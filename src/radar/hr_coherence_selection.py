"""Single definition of the experimental HR-power/coherent-sum selector.

The spatial axes must describe simultaneous channels. Unit-IQ coherent gain
depends on channel phase offsets and direction; it is NOT calibrated SNR,
chest localisation, or evidence of cardiac identity.
"""
from __future__ import annotations

import numpy as np

from .hr_phase_score import hr_phase_power

SELECTOR_VERSION = "hr_phase_power_top8_unit_iq_v1"


def select_hr_coherence_range(cube: np.ndarray, fs: float,
                              hr_band: tuple[float, float],
                              top_k: int = 8) -> int:
    """Select one range from all ranges using identical rules in every caller.

    Input is (F,D,A,R). HR power is computed per cell by the exact conjugate
    increment/Hann routine, then averaged over (D,A). Score the top K ranges
    by log1p(power / all-range median) times mean unit-IQ coherent gain.
    All-zero HR power falls back to the maximum mean IQ-power range.
    """
    z = np.asarray(cube, dtype=np.complex64)
    if z.ndim != 4 or min(z.shape) < 1 or z.shape[0] < 8:
        raise ValueError("Expected nonempty (F,D,A,R) cube with at least eight frames")
    if not np.isfinite(z).all():
        raise ValueError("IQ must be finite")
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    profile = hr_phase_power(z, fs, hr_band).mean(axis=(0, 1))
    wide = z.astype(np.complex128)
    if not np.any(profile > 0):
        return int(np.argmax(np.mean(np.abs(wide) ** 2, axis=(0, 1, 2))))
    # Stable ties prefer the lower range index.
    candidates = np.argsort(-profile, kind="stable")[:min(top_k, len(profile))]
    unit = wide / np.maximum(np.abs(wide), 1e-12)
    coherent_gain = np.mean(np.abs(np.mean(unit, axis=(1, 2))), axis=0)
    scores = np.log1p(profile[candidates] / (np.median(profile) + 1e-12)) * coherent_gain[candidates]
    return int(candidates[int(np.argmax(scores))])
