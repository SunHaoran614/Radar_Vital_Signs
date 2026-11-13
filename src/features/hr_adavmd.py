"""Heart-rate-adaptive VMD decomposition.

This module contains the innovation block after EDACM. It decomposes the fused
radar micro-motion signal with physiology-guided VMD initialization, scores each
mode by heart-rate-band evidence, and applies adaptive mode weighting. The output
keeps a fixed ``(K, frames)`` shape so different backbones can use the same input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.features.edacm import zscore_1d

DEFAULT_VMD_K = 7
DEFAULT_VMD_ALPHA = 2000.0
DEFAULT_VMD_MAX_ITER = 120
DEFAULT_VMD_TOL = 1e-5
DEFAULT_SAMPLING_RATE_HZ = 20.0
DEFAULT_HR_BAND_HZ = (0.75, 2.5)
DEFAULT_RESP_BAND_HZ = (0.1, 0.6)
DECOMPOSITION_ALGORITHM_VERSION = "positive_rfft_gauss_seidel_amplitude_v2"

def vmd_decompose(
    signal: np.ndarray,
    k: int = DEFAULT_VMD_K,
    alpha: float = DEFAULT_VMD_ALPHA,
    max_iter: int = DEFAULT_VMD_MAX_ITER,
    tol: float = DEFAULT_VMD_TOL,
    init_omega: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Stable positive-frequency VMD update for one real-valued signal.

    Updates use the current sum of modes (Gauss-Seidel). A real FFT keeps only
    independent positive frequencies; inverse rFFT restores conjugate symmetry.
    Mode amplitudes are retained, since per-mode z-scoring destroys the
    decomposition's reconstruction semantics.
    """
    x = zscore_1d(signal)
    n = int(x.size)
    if n == 0:
        return np.empty((k, 0), dtype=np.float32)
    if np.std(x) < 1e-8:
        return np.zeros((k, n), dtype=np.float32)

    freqs = np.fft.rfftfreq(n).astype(np.float64)
    spectrum = np.fft.rfft(x).astype(np.complex128)

    u_hat = np.zeros((k, freqs.size), dtype=np.complex128)
    if init_omega is None:
        omega = np.linspace(0.0, 0.5, k + 2, dtype=np.float64)[1:-1]
    else:
        omega = np.asarray(init_omega, dtype=np.float64).reshape(-1)
        if omega.size != k:
            raise ValueError(f"init_omega must contain {k} value(s)")
        omega = np.clip(omega, 0.0, 0.5).astype(np.float64)

    for _ in range(max_iter):
        previous = u_hat.copy()
        sum_modes = np.sum(u_hat, axis=0)
        for mode_idx in range(k):
            residual = spectrum - (sum_modes - u_hat[mode_idx])
            denom = 1.0 + alpha * (freqs - omega[mode_idx]) ** 2
            update = residual / denom
            if not np.all(np.isfinite(update)):
                raise FloatingPointError("Nonfinite VMD mode update")
            sum_modes += update - u_hat[mode_idx]
            u_hat[mode_idx] = update
            power = np.abs(update) ** 2
            power_sum = float(np.sum(power))
            if np.isfinite(power_sum) and power_sum > 1e-12:
                new_omega = float(np.sum(freqs * power) / power_sum)
                if np.isfinite(new_omega):
                    omega[mode_idx] = float(np.clip(new_omega, 0.0, 0.5))
        diff = np.linalg.norm(u_hat - previous) / (np.linalg.norm(previous) + 1e-12)
        if not np.isfinite(diff):
            raise FloatingPointError("Nonfinite VMD convergence metric")
        if diff < tol:
            break

    modes = np.fft.irfft(u_hat, n=n, axis=1).astype(np.float32)
    if not np.all(np.isfinite(modes)):
        raise FloatingPointError("Nonfinite VMD output")
    return modes


def hr_adavmd_initial_omega(
    k: int,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    init_profile: str = "legacy",
) -> np.ndarray:
    """Physiology-guided VMD center-frequency initialization."""
    if init_profile == "expanded_3p5":
        # Explicit experimental arm; merely covering 3.2 Hz does not prove
        # better decomposition on real FTU recordings.
        base_hz = np.asarray([0.2, 0.45, 0.8, 1.2, 1.8, 2.5, 3.2, 4.2, 5.0], dtype=np.float32)
    elif init_profile == "legacy":
        base_hz = np.asarray([0.2, 0.45, 0.8, 1.1, 1.45, 1.9, 2.4, 3.2, 4.2], dtype=np.float32)
    else:
        raise ValueError(f"Unknown VMD initialization profile: {init_profile}")
    if k <= base_hz.size:
        centers_hz = base_hz[:k]
    else:
        extra = np.linspace(float(base_hz[-1]), sampling_rate_hz / 2.0, k - base_hz.size + 2, dtype=np.float32)[1:-1]
        centers_hz = np.concatenate([base_hz, extra])
    return np.clip(centers_hz / float(sampling_rate_hz), 0.0, 0.5).astype(np.float32)


def band_energy(power: np.ndarray, freq_hz: np.ndarray, band_hz: Tuple[float, float]) -> float:
    mask = (freq_hz >= float(band_hz[0])) & (freq_hz <= float(band_hz[1]))
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def hr_adavmd_mode_scores(
    modes: np.ndarray,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    hr_band_hz: Tuple[float, float] = DEFAULT_HR_BAND_HZ,
    resp_band_hz: Tuple[float, float] = DEFAULT_RESP_BAND_HZ,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    time_modes = np.asarray(modes, dtype=np.float32)
    n = int(time_modes.shape[-1])
    if n <= 1:
        weights = np.ones(time_modes.shape[0], dtype=np.float32)
        scores = np.ones(time_modes.shape[0], dtype=np.float32)
        return weights, scores, {
            "hr_band_hz": list(hr_band_hz),
            "resp_band_hz": list(resp_band_hz),
            "mode_scores": scores.tolist(),
            "mode_weights": weights.tolist(),
        }

    freq_hz = np.fft.rfftfreq(n, d=1.0 / float(sampling_rate_hz)).astype(np.float32)
    raw_scores: List[float] = []
    diagnostics: List[Dict[str, float]] = []
    for mode in time_modes:
        spectrum = np.fft.rfft(zscore_1d(mode) * np.hanning(n).astype(np.float32))
        power = np.abs(spectrum).astype(np.float32) ** 2
        total = float(np.sum(power)) + 1e-12
        hr_energy = band_energy(power, freq_hz, hr_band_hz)
        resp_energy = band_energy(power, freq_hz, resp_band_hz)
        high_energy = band_energy(power, freq_hz, (float(hr_band_hz[1]), float(freq_hz[-1]) if freq_hz.size else hr_band_hz[1]))
        hr_mask = (freq_hz >= float(hr_band_hz[0])) & (freq_hz <= float(hr_band_hz[1]))
        if np.any(hr_mask):
            hr_power = power[hr_mask]
            peak_sharpness = float(np.max(hr_power) / (np.mean(hr_power) + 1e-12))
        else:
            peak_sharpness = 0.0
        hr_ratio = hr_energy / total
        resp_ratio = resp_energy / total
        high_ratio = high_energy / total
        score = 0.70 * hr_ratio + 0.20 * np.tanh(peak_sharpness / 5.0) - 0.25 * resp_ratio - 0.10 * high_ratio
        raw_scores.append(float(score))
        diagnostics.append(
            {
                "hr_ratio": float(hr_ratio),
                "resp_ratio": float(resp_ratio),
                "high_ratio": float(high_ratio),
                "peak_sharpness": float(peak_sharpness),
                "score": float(score),
            }
        )

    scores = np.asarray(raw_scores, dtype=np.float32)
    scores = scores - float(np.min(scores))
    if float(np.max(scores)) > 1e-6:
        scores = scores / float(np.max(scores))
    else:
        scores = np.ones_like(scores, dtype=np.float32)
    weights = 0.5 + 0.5 * scores
    return weights.astype(np.float32), scores.astype(np.float32), {
        "hr_band_hz": list(hr_band_hz),
        "resp_band_hz": list(resp_band_hz),
        "mode_scores": [float(x) for x in scores.tolist()],
        "mode_weights": [float(x) for x in weights.tolist()],
        "mode_diagnostics": diagnostics,
    }


def hr_adavmd_decompose(
    signal: np.ndarray,
    k: int = DEFAULT_VMD_K,
    alpha: float = DEFAULT_VMD_ALPHA,
    max_iter: int = DEFAULT_VMD_MAX_ITER,
    tol: float = DEFAULT_VMD_TOL,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    hr_band_hz: Tuple[float, float] = DEFAULT_HR_BAND_HZ,
    init_profile: str = "legacy",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Heart-rate-aware adaptive VMD for radar micro-motion signals."""
    init_omega = hr_adavmd_initial_omega(
        k=k, sampling_rate_hz=sampling_rate_hz, init_profile=init_profile)
    modes = vmd_decompose(
        signal=signal,
        k=k,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        init_omega=init_omega,
    )
    weights, scores, score_meta = hr_adavmd_mode_scores(
        modes,
        sampling_rate_hz=sampling_rate_hz,
        hr_band_hz=hr_band_hz,
    )
    weighted_modes = (modes * weights[:, None]).astype(np.float32)
    meta = {
        "decomposition_method": "HR-AdaVMD",
        "decomposition_algorithm_version": DECOMPOSITION_ALGORITHM_VERSION,
        "vmd_init_profile": init_profile,
        "hr_adavmd_components": [
            "physiology_guided_frequency_initialization",
            "heart_band_mode_scoring",
            "adaptive_mode_weighting",
        ],
        "hr_adavmd_init_omega": [float(x) for x in init_omega.tolist()],
        "hr_adavmd_init_center_hz": [float(x * sampling_rate_hz) for x in init_omega.tolist()],
        **score_meta,
        "selected_mode_count": int(k),
    }
    return weighted_modes, meta
