"""EDACM phase representation for radar vital-sign windows.

This module owns the first stage of the proposed pipeline. It receives one
complex RDA window shaped as ``(frames, doppler, angle, range)``, selects stable
vital-sign target bins, extracts EDACM phase traces, and fuses them into one
normalized micro-motion signal. Keeping this code separate makes the target
selection and phase construction easy to review independently from HR-AdaVMD.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

DEFAULT_EDACM_TOP_BINS = 3
DEFAULT_EDACM_CANDIDATE_MULTIPLIER = 8
DEFAULT_HR_BAND_HZ = (0.75, 2.5)
DEFAULT_RDA_STABILITY_SEGMENTS = 4
DEFAULT_SAMPLING_RATE_HZ = 20.0

def detrend_linear(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).reshape(-1)
    if y.size <= 1:
        return y - np.nanmean(y)
    t = np.linspace(-1.0, 1.0, y.size, dtype=np.float32)
    valid = np.isfinite(y)
    if int(valid.sum()) < 2:
        return np.nan_to_num(y - np.nanmean(y), nan=0.0).astype(np.float32)
    slope, intercept = np.polyfit(t[valid], y[valid], deg=1)
    out = y - (slope * t + intercept).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    if y.size == 0 or not np.isfinite(y).any():
        return np.zeros_like(y, dtype=np.float32)
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    if np.isfinite(std) and std > 1e-6:
        y = (y - mean) / std
    else:
        y = y - mean
    y[~np.isfinite(y)] = 0.0
    return y.astype(np.float32)


def edacm_phase(z: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    series = np.asarray(z, dtype=np.complex64).reshape(-1)
    i = np.real(series).astype(np.float32)
    q = np.imag(series).astype(np.float32)
    di = np.diff(i, prepend=i[:1])
    dq = np.diff(q, prepend=q[:1])
    denom = i * i + q * q + np.float32(eps)
    delta_phase = (i * dq - q * di) / denom
    phase = np.cumsum(delta_phase, dtype=np.float32)
    return detrend_linear(phase)


def minmax_score(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.min(x[finite]))
    hi = float(np.max(x[finite]))
    if hi - lo <= 1e-8:
        return np.ones_like(x, dtype=np.float32)
    out = (x - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def phase_stability_score(phase: np.ndarray) -> float:
    y = zscore_1d(phase)
    if y.size < 4:
        return 0.0
    diff = np.diff(y)
    roughness = float(np.nanstd(diff))
    if not np.isfinite(roughness):
        return 0.0
    return float(1.0 / (1.0 + roughness))


def hr_band_peak_score(
    phase: np.ndarray,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    band_hz: Tuple[float, float] = DEFAULT_HR_BAND_HZ,
) -> float:
    y = zscore_1d(phase)
    n = int(y.size)
    if n < 8:
        return 0.0
    window = np.hanning(n).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(y * window)).astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sampling_rate_hz)).astype(np.float32)
    valid = (freqs >= float(band_hz[0])) & (freqs <= float(band_hz[1]))
    if not valid.any():
        return 0.0
    total = float(np.sum(spectrum[1:]) + 1e-8)
    band = spectrum[valid]
    peak = float(np.max(band)) if band.size else 0.0
    band_energy = float(np.sum(band))
    if not np.isfinite(peak) or not np.isfinite(band_energy):
        return 0.0
    peak_concentration = peak / (float(np.mean(band)) + 1e-8) if band.size else 0.0
    band_ratio = band_energy / total
    return float(np.log1p(max(0.0, peak_concentration)) * max(0.0, band_ratio))


def segment_peak_bins(
    radar: np.ndarray,
    segments: int = DEFAULT_RDA_STABILITY_SEGMENTS,
) -> np.ndarray:
    n_frames = int(radar.shape[0])
    n_segments = min(max(1, int(segments)), max(1, n_frames))
    peaks: List[np.ndarray] = []
    for idx in range(n_segments):
        start = int(round(idx * n_frames / n_segments))
        end = int(round((idx + 1) * n_frames / n_segments))
        if end <= start:
            continue
        energy = np.mean(np.abs(radar[start:end]) ** 2, axis=0)
        peaks.append(np.asarray(np.unravel_index(int(np.argmax(energy)), energy.shape), dtype=np.float32))
    if not peaks:
        energy = np.mean(np.abs(radar) ** 2, axis=0)
        peaks.append(np.asarray(np.unravel_index(int(np.argmax(energy)), energy.shape), dtype=np.float32))
    return np.stack(peaks, axis=0)


def spatial_consistency_scores(candidate_bins: np.ndarray, peak_bins: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidate_bins, dtype=np.float32)
    peaks = np.asarray(peak_bins, dtype=np.float32)
    if candidates.size == 0 or peaks.size == 0:
        return np.zeros((candidates.shape[0],), dtype=np.float32)
    axis_scale = np.maximum(np.ptp(peaks, axis=0), 1.0).astype(np.float32)
    scores = []
    for candidate in candidates:
        dist = np.linalg.norm((peaks - candidate[None, :]) / axis_scale[None, :], axis=1)
        scores.append(float(np.mean(np.exp(-dist))))
    return np.asarray(scores, dtype=np.float32)


def select_target_bin_indices(
    radar: np.ndarray,
    top_bins: int,
    hr_band_hz: Tuple[float, float] = DEFAULT_HR_BAND_HZ,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    energy = np.mean(np.abs(radar) ** 2, axis=0)
    flat = energy.reshape(-1)
    count = min(max(1, int(top_bins)), int(flat.size))
    candidate_count = min(
        int(flat.size),
        max(count, count * DEFAULT_EDACM_CANDIDATE_MULTIPLIER),
    )
    candidate_flat = np.argpartition(flat, -candidate_count)[-candidate_count:]
    candidate_multi = np.asarray(np.unravel_index(candidate_flat, energy.shape)).T.astype(np.int32)

    phase_stability = []
    hr_peak = []
    for d_idx, a_idx, r_idx in candidate_multi:
        phase = edacm_phase(radar[:, int(d_idx), int(a_idx), int(r_idx)])
        phase_stability.append(phase_stability_score(phase))
        hr_peak.append(hr_band_peak_score(phase, band_hz=hr_band_hz))

    peak_bins = segment_peak_bins(radar)
    energy_scores = minmax_score(np.log1p(flat[candidate_flat]))
    phase_scores = minmax_score(np.asarray(phase_stability, dtype=np.float32))
    hr_scores = minmax_score(np.asarray(hr_peak, dtype=np.float32))
    spatial_scores = minmax_score(spatial_consistency_scores(candidate_multi, peak_bins))
    combined = (
        0.35 * energy_scores
        + 0.25 * phase_scores
        + 0.25 * hr_scores
        + 0.15 * spatial_scores
    ).astype(np.float32)

    order = np.argsort(combined)[::-1][:count]
    selected_multi = candidate_multi[order].astype(np.int32)
    selected_scores = combined[order].astype(np.float32)
    selected_energy = flat[candidate_flat[order]].astype(np.float32)
    weights = selected_scores.copy()
    if float(np.sum(weights)) <= 1e-12:
        weights = np.full(count, 1.0 / count, dtype=np.float32)
    else:
        weights = weights / np.sum(weights)
    score_meta = {
        "target_selection_method": "rda_vital_sign_stability_score",
        "candidate_bins": int(candidate_count),
        "score_weights": {
            "energy": 0.35,
            "phase_stability": 0.25,
            "hr_band_peak": 0.25,
            "spatial_consistency": 0.15,
        },
        "hr_band_hz": [float(hr_band_hz[0]), float(hr_band_hz[1])],
        "spatial_stability_segments": int(DEFAULT_RDA_STABILITY_SEGMENTS),
        "segment_peak_bins_dar": peak_bins.astype(np.int32).tolist(),
        "target_bin_scores": [float(x) for x in selected_scores.tolist()],
        "target_bin_energy": [float(x) for x in selected_energy.tolist()],
        "target_bin_energy_score": [float(x) for x in energy_scores[order].tolist()],
        "target_bin_phase_stability_score": [float(x) for x in phase_scores[order].tolist()],
        "target_bin_hr_band_peak_score": [float(x) for x in hr_scores[order].tolist()],
        "target_bin_spatial_consistency_score": [float(x) for x in spatial_scores[order].tolist()],
        "rda_spatial_confidence": float(np.sum(selected_scores * weights)),
    }
    return selected_multi, weights.astype(np.float32), score_meta


def target_edacm_signal(
    radar: np.ndarray,
    top_bins: int = DEFAULT_EDACM_TOP_BINS,
    hr_band_hz: Tuple[float, float] = DEFAULT_HR_BAND_HZ,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    selected_bins, weights, score_meta = select_target_bin_indices(
        radar, top_bins=top_bins, hr_band_hz=hr_band_hz)
    phases = []
    for d_idx, a_idx, r_idx in selected_bins:
        z = radar[:, int(d_idx), int(a_idx), int(r_idx)]
        phases.append(edacm_phase(z))
    stacked = np.stack(phases, axis=0).astype(np.float32)
    fused = np.sum(stacked * weights[:, None], axis=0)
    fused = zscore_1d(fused)
    meta = {
        "phase_method": "EDACM",
        "edacm_top_bins": int(top_bins),
        "target_bins_dar": selected_bins.tolist(),
        "target_bin_weights": [float(x) for x in weights.tolist()],
        "phase_preprocess": "linear_detrend_zscore",
        **score_meta,
    }
    return fused.astype(np.float32), meta

