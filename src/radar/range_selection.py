"""Range selection stage: choose which range bins feed downstream.

After localization has focused the spatial region, range selection picks the
range axis subset. Because the (F, D, A, R) contract must be preserved, selection
is done by *zeroing* the rejected range bins (a soft/hard mask along axis 3), so
the downstream EDACM phase extraction still sees a full cube but ignores off-target
ranges. For PhysDrive the cube is already cropped to 8 range bins; this stage
re-weights *within* those 8 rather than expanding them.

Candidate matrix rows: R0 (global energy crop, current), R1 (peak range),
R2 (energy-weighted center), R3 (HR-band-weighted center -- the important one).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .pipeline import register_stage, DEFAULT_FS
from .hr_phase_score import hr_phase_power
from .numerics import magnitude_squared


def _complex_safe(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.complex64)


def _range_profile(cube: np.ndarray, mode: str = "energy",
                   fs: float = DEFAULT_FS,
                   hr_band: tuple[float, float] = (0.75, 3.5)) -> np.ndarray:
    """Scalar per-range score collapsed over (F, D, A)."""
    mag2 = np.mean(magnitude_squared(cube), axis=(0, 1, 2))  # (R,)
    if mode == "energy":
        return mag2
    if mode == "hr_band":
        f = cube.shape[0]
        if f < 8:
            return mag2
        return hr_phase_power(cube, fs, hr_band).mean(axis=(0, 1))
    return mag2


def _mask_range(cube: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Zero out range bins not in ``keep`` (boolean/int index over axis 3)."""
    cube = _complex_safe(cube)
    mask = np.zeros(cube.shape[3], dtype=np.float32)
    mask[keep] = 1.0
    return cube * mask[None, None, None, :]


def range_global_energy(cube: np.ndarray, half_width: int = 4, **_: Any) -> np.ndarray:
    """R0: current baseline -- global energy center +- half_width bins."""
    cube = _complex_safe(cube)
    r = cube.shape[3]
    profile = _range_profile(cube, mode="energy")
    center = int(np.argmax(profile))
    lo = max(0, center - half_width)
    hi = min(r, center + half_width + 1)
    keep = np.arange(lo, hi)
    return _mask_range(cube, keep)


def range_peak(cube: np.ndarray, half_width: int = 1, **_: Any) -> np.ndarray:
    """R1: keep only the single peak-energy range (+- half_width)."""
    cube = _complex_safe(cube)
    r = cube.shape[3]
    profile = _range_profile(cube, mode="energy")
    center = int(np.argmax(profile))
    lo = max(0, center - half_width)
    hi = min(r, center + half_width + 1)
    return _mask_range(cube, np.arange(lo, hi))


def range_weighted_center(cube: np.ndarray, half_width: int = 4, **_: Any) -> np.ndarray:
    """R2: energy-weighted centroid, keep +- half_width around it."""
    cube = _complex_safe(cube)
    r = cube.shape[3]
    profile = _range_profile(cube, mode="energy")
    if profile.sum() <= 0:
        return cube
    center = int(round(float(np.sum(np.arange(r) * profile) / profile.sum())))
    lo = max(0, center - half_width)
    hi = min(r, center + half_width + 1)
    return _mask_range(cube, np.arange(lo, hi))


def range_hr_band(cube: np.ndarray, half_width: int = 2,
                  fs: float = DEFAULT_FS,
                  hr_band: tuple[float, float] = (0.75, 3.5),
                  **_: Any) -> np.ndarray:
    """R3: HR-band-weighted range center -- keep the range most likely to hold the heartbeat.

    This directly attacks the "total energy max != heartbeat max" issue: car
    seats, clothing and structure can dominate total amplitude but carry no HR
    micro-motion. Selecting on HR-band phase energy instead focuses the cube on
    the cardioballistic reflection.
    """
    cube = _complex_safe(cube)
    r = cube.shape[3]
    profile = _range_profile(cube, mode="hr_band", fs=fs, hr_band=hr_band)
    if profile.sum() <= 0:
        # fall back to energy weighting if no HR-band evidence
        profile = _range_profile(cube, mode="energy", fs=fs)
    total = profile.sum()
    if total <= 0:
        return cube
    center = int(round(float(np.sum(np.arange(r) * profile) / total)))
    lo = max(0, center - half_width)
    hi = min(r, center + half_width + 1)
    return _mask_range(cube, np.arange(lo, hi))


def range_hr_band_coherence(cube: np.ndarray, half_width: int = 2,
                            top_k: int = 8, fs: float = DEFAULT_FS,
                            hr_band: tuple[float, float] = (0.75, 3.5),
                            **_: Any) -> np.ndarray:
    """HR-band range candidate penalised by cross-spatial IQ incoherence.

    HR-band power alone can select an isolated noisy antenna/range cell.  Keep
    only the top ``top_k`` HR-power ranges, then score each by
    ``log1p(power / median_power) * unit-circle coherence`` over the spatial
    (D,A) axes.  This is radar-only and preserves the cube shape contract.
    """
    from .hr_coherence_selection import select_hr_coherence_range

    cube = _complex_safe(cube)
    r = cube.shape[3]
    center = select_hr_coherence_range(cube, fs, hr_band, top_k)
    lo = max(0, int(center) - half_width)
    hi = min(r, int(center) + half_width + 1)
    return _mask_range(cube, np.arange(lo, hi))


def register_range_selection_methods() -> None:
    register_stage("range_selection", "global_energy", range_global_energy)
    register_stage("range_selection", "peak", range_peak)
    register_stage("range_selection", "weighted_center", range_weighted_center)
    register_stage("range_selection", "hr_band", range_hr_band)
    register_stage("range_selection", "hr_band_coherence", range_hr_band_coherence)
