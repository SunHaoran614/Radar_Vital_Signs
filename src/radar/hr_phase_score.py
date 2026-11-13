"""Label-free slow-time phase power for RDA localization candidates."""

from __future__ import annotations

import numpy as np


def hr_phase_power(cube: np.ndarray, fs: float,
                   hr_band: tuple[float, float]) -> np.ndarray:
    """Return heart-band phase power per (D,A,R) cell.

    Adjacent phase increments use the exact conjugate product. Weak-IQ frames
    are masked so an almost-zero complex sample cannot dominate the score.
    """
    f = cube.shape[0]
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    low, high = map(float, hr_band)
    if not (0 < low < high < fs / 2):
        raise ValueError("hr_band must lie inside (0, Nyquist)")
    z = np.asarray(cube, dtype=np.complex128).reshape(f, -1)
    power = np.abs(z) ** 2
    floor = 1e-4 * np.median(power, axis=0, keepdims=True)
    valid = (power[1:] > floor) & (power[:-1] > floor)
    increments = np.angle(z[1:] * np.conj(z[:-1]))
    increments = np.where(valid, increments, 0.0)
    phase = np.concatenate((np.zeros_like(increments[:1]),
                            np.cumsum(increments, axis=0)), axis=0)
    phase -= phase.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(phase * np.hanning(f)[:, None], axis=0)) ** 2
    freqs = np.fft.rfftfreq(f, d=1.0 / fs)
    band = (freqs >= low) & (freqs <= high)
    return spectrum[band].sum(axis=0).reshape(cube.shape[1:])
