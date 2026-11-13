"""Exact repeated-frame support and explicit offline interpolation candidates.

The first nonzero frame of a constant run is retained as an anchor. Repeated
frames are observations with unknown freshness, not a certificate of a lost
packet. This module never compresses time or asserts recovered measurements.
"""
from __future__ import annotations

import numpy as np


def frame_support(cube: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(cube)
    if x.ndim < 2 or len(x) < 2 or not np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite complex [frames, ...] with at least two frames')
    axes = tuple(range(1, x.ndim))
    zero = np.all(x == 0, axis=axes)
    repeated = np.r_[False, np.all(x[1:] == x[:-1], axis=axes)]
    anchor = ~zero & ~repeated
    return dict(zero=zero, repeated=repeated, anchor=anchor)


def interpolate_frame_support(cube: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Offline complex linear candidate on the unchanged original frame grid.

    Valid anchors remain exactly unchanged. Leading/trailing gaps hold the
    closest anchor; an all-missing record remains unchanged. Long gaps cannot
    be interpreted as restored physiological information.
    """
    x = np.asarray(cube)
    valid = np.asarray(anchor)
    if x.ndim < 2 or valid.dtype != bool or valid.shape != (len(x),):
        raise ValueError('Expected one boolean anchor per original frame')
    if not np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite complex data')
    result = x.copy()
    if valid.any() and not valid.all():
        indices = np.arange(len(x))
        flat = result.reshape(len(x), -1)
        for cell in range(flat.shape[1]):
            values = flat[:, cell].copy()
            flat[~valid, cell] = (np.interp(indices[~valid], indices[valid], values[valid].real)
                                 + 1j*np.interp(indices[~valid], indices[valid], values[valid].imag))
    return result
