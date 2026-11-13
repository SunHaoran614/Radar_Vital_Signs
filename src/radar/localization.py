"""Localization stage: find the spatial region containing the subject.

The cube is ``(F, D, A, R)``. Localization selects a spatial *focus* (a set of
range/angle/doppler coordinates or a soft mask) so downstream range-selection and
beamforming operate on the right place. Because we must preserve the (F, D, A, R)
shape contract, localization here returns a *weighted* cube: it multiplies the
cube by a spatial confidence mask (1.0 at the located region, attenuating
elsewhere) rather than literally cropping. Cropping/reindexing is done later by
``range_selection`` if desired.

Why this matters (see docs/frontend.md, section 4):
    Weak PhysDrive prediction motivates checking whether spatial selection
    preserves cardiac micro-motion. Total energy may reflect clutter or body
    motion, so it cannot by itself establish the correct vital-sign region.

Candidate matrix rows: L0 (energy), L1 (spatial consistency), L2 (HR-band
weighted), L3 (temporal consistency).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .pipeline import register_stage, DEFAULT_FS
from .hr_phase_score import hr_phase_power
from .numerics import magnitude_squared


def _complex_safe(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.complex64)


def _spatial_energy(cube: np.ndarray) -> np.ndarray:
    """Mean squared magnitude over the frame axis -> (D, A, R)."""
    return np.mean(magnitude_squared(cube), axis=0)


def localization_energy(cube: np.ndarray, **_: Any) -> np.ndarray:
    """L0: locate the single (D,A,R) cell with max total-energy; focus a soft window there.

    This is the *current implicit baseline* (global energy max). Implemented as a
    Gaussian-weighted mask so it is differentiable from the other methods while
    still returning a full cube.
    """
    cube = _complex_safe(cube)
    energy = _spatial_energy(cube)
    peak = np.unravel_index(int(np.argmax(energy)), energy.shape)
    mask = _gaussian_mask(energy.shape, peak, sigma=1.0)
    return cube * mask[None, ...]


def localization_spatial_consistency(cube: np.ndarray, sigma: float = 1.5, **_: Any) -> np.ndarray:
    """L1: peak per-frame, then keep the spatially *consistent* region across frames.

    Computes per-frame spatial peaks, averages their locations, and builds a
    confidence mask around the temporally-consistent centroid.
    """
    cube = _complex_safe(cube)
    f = cube.shape[0]
    energy = magnitude_squared(cube)  # (F, D, A, R)
    peaks = [np.unravel_index(int(np.argmax(energy[t])), energy[t].shape) for t in range(f)]
    peaks = np.asarray(peaks, dtype=np.float32)  # (F, 3) -> (d, a, r)
    centroid = peaks.mean(axis=0)
    center = tuple(min(max(int(round(value)), 0), size - 1)
                   for value, size in zip(centroid, cube.shape[1:]))
    mask = _gaussian_mask(cube.shape[1:], center, sigma=sigma)
    return cube * mask[None, ...]


def localization_hr_band(cube: np.ndarray, hr_band: tuple = (0.75, 3.5),
                         fs: float = DEFAULT_FS, **_: Any) -> np.ndarray:
    """L2: locate the spatial region with the most *HR-band* micro-motion energy.

    Instead of total energy, project each (D,A,R) slow-time signal to its
    heart-rate-band spectral energy and localize on THAT. This is the key fix for
    the "total energy max != heartbeat max" problem on PhysDrive / in-vehicle.
    """
    cube = _complex_safe(cube)
    f, d, a, r = cube.shape
    if f < 8:
        return localization_energy(cube)
    hr_energy = hr_phase_power(cube, fs, hr_band)
    peak = np.unravel_index(int(np.argmax(hr_energy)), hr_energy.shape)
    mask = _gaussian_mask((d, a, r), peak, sigma=1.0)
    return cube * mask[None, ...]


def localization_temporal_consistency(cube: np.ndarray, sigma: float = 1.5,
                                      fs: float = DEFAULT_FS, **_: Any) -> np.ndarray:
    """L3: track the loudest spatial cell across frames, mask the consistent track.

    Averages per-frame maximum-amplitude locations and builds a soft mask around
    their temporal centroid. This is an amplitude heuristic, not a heartbeat
    tracker; its effectiveness must be checked with label-free prediction.
    """
    cube = _complex_safe(cube)
    f, d, a, r = cube.shape
    if f < 8:
        return localization_spatial_consistency(cube, sigma=sigma)
    # Per-frame spatial amplitude proxy; retain the range coordinate.
    per_frame = np.abs(cube)  # (F, D, A, R)
    peaks = [np.unravel_index(int(np.argmax(per_frame[t])), per_frame[t].shape) for t in range(f)]
    peaks = np.asarray(peaks, dtype=np.float32)  # (F, 3) -> (d, a, r)
    centroid = peaks.mean(axis=0)
    center = tuple(min(max(int(round(value)), 0), size - 1)
                   for value, size in zip(centroid, (d, a, r)))
    mask = _gaussian_mask((d, a, r), center, sigma=sigma)
    return cube * mask[None, ...]


def _gaussian_mask(shape, center, sigma: float = 1.0) -> np.ndarray:
    """Soft Gaussian confidence mask over a (D, A, R) spatial volume."""
    d, a, r = shape
    cd, ca, cr = center
    z = np.zeros(shape, dtype=np.float32)
    # limit support to keep it cheap and local
    rad = max(2, int(round(3 * sigma)))
    for dd in range(max(0, cd - rad), min(d, cd + rad + 1)):
        for aa in range(max(0, ca - rad), min(a, ca + rad + 1)):
            for rr in range(max(0, cr - rad), min(r, cr + rad + 1)):
                val = np.exp(-((dd - cd) ** 2 + (aa - ca) ** 2 + (rr - cr) ** 2) / (2 * sigma**2))
                z[dd, aa, rr] = val
    # normalise so the located center keeps unit gain, off-region is attenuated
    if z.max() > 0:
        z = z / z.max()
    return z


def register_localization_methods() -> None:
    register_stage("localization", "energy", localization_energy)
    register_stage("localization", "spatial_consistency", localization_spatial_consistency)
    register_stage("localization", "hr_band", localization_hr_band)
    register_stage("localization", "temporal_consistency", localization_temporal_consistency)
