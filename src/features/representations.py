"""Feature representation assembly for training windows.

This module is the single routing point for converting a complex RDA window into
model-ready tensors. The proposed path is intentionally explicit:
EDACM phase representation -> HR-AdaVMD -> time/frequency features. Simpler
representations are retained here as baselines for comparison experiments.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from src.features.edacm import DEFAULT_HR_BAND_HZ, target_edacm_signal, zscore_1d
from src.features.hr_adavmd import (
    DEFAULT_VMD_ALPHA,
    DEFAULT_VMD_K,
    DEFAULT_VMD_MAX_ITER,
    DEFAULT_VMD_TOL,
    hr_adavmd_decompose,
)

DEFAULT_SAMPLING_RATE_HZ = 20.0
DEFAULT_RDA_REPRESENTATION = "log_magnitude"
METHOD_PIPELINE = (
    "radar_rda_features",
    "edacm_phase_representation",
    "hr_adavmd_decomposition",
    "time_frequency_feature_construction",
    "cycleformer_period_token_regression",
    "pseudo_label_temporal_domain_adaptation",
    "heart_rate_regression",
)
INNOVATION_MODULES = (
    "EDACM target phase representation",
    "HR-AdaVMD radar micro-motion decomposition",
    "CycleFormer physiological period-token modeling",
    "Pseudo-label temporal adaptation for cross-dataset heart-rate estimation",
)
DOMAIN_ADAPTATION_METHOD = "Pseudo-label adaptation + temporal confidence weighting"

def frequency_features(
    modes: np.ndarray,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    time_modes = np.asarray(modes, dtype=np.float32)
    if time_modes.ndim == 1:
        time_modes = time_modes[None, :]
    n = int(time_modes.shape[-1])
    if n == 0:
        return (
            np.empty((*time_modes.shape[:-1], 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            {
                "frequency_feature": "hann_rfft_log_magnitude",
                "sampling_rate_hz": float(sampling_rate_hz),
                "frequency_bins": 0,
                "frequency_resolution_hz": None,
            },
        )

    window = np.hanning(n).astype(np.float32)
    spectrum = np.fft.rfft(time_modes * window[None, :], axis=-1)
    x_freq = np.log1p(np.abs(spectrum)).astype(np.float32)
    x_freq = np.stack([zscore_1d(row) for row in x_freq], axis=0).astype(np.float32)
    freq_hz = np.fft.rfftfreq(n, d=1.0 / float(sampling_rate_hz)).astype(np.float32)
    meta = {
        "frequency_feature": "hann_rfft_log_magnitude",
        "sampling_rate_hz": float(sampling_rate_hz),
        "frequency_bins": int(freq_hz.size),
        "frequency_resolution_hz": float(freq_hz[1] - freq_hz[0]) if freq_hz.size > 1 else None,
        "frequency_range_hz": [float(freq_hz[0]), float(freq_hz[-1])] if freq_hz.size else [],
    }
    return x_freq, freq_hz, meta


def rda_log_magnitude(radar: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(radar)).astype(np.float32)


def radar_to_feature_bundle(
    radar: np.ndarray,
    representation: str,
    hr_band_hz: Tuple[float, float] = DEFAULT_HR_BAND_HZ,
    vmd_init_profile: str = "legacy",
    include_rda: bool = True,
) -> Dict[str, Any]:
    if representation == "real_imag":
        x = np.stack([np.real(radar), np.imag(radar)], axis=0).astype(np.float32)
    elif representation == "magnitude":
        x = np.abs(radar).astype(np.float32)
    elif representation == "log_magnitude":
        x = np.log1p(np.abs(radar)).astype(np.float32)
    elif representation == "target_edacm":
        phase, phase_meta = target_edacm_signal(radar, hr_band_hz=hr_band_hz)
        x = phase[None, :].astype(np.float32)
        return {"x": x, "meta": phase_meta}
    elif representation in {"proposed", "target_edacm_vmd"}:
        phase, phase_meta = target_edacm_signal(radar, hr_band_hz=hr_band_hz)
        x_time, hr_adavmd_meta = hr_adavmd_decompose(
            phase, k=DEFAULT_VMD_K, hr_band_hz=hr_band_hz,
            init_profile=vmd_init_profile)
        x_freq, freq_hz, freq_meta = frequency_features(x_time)
        x_rda = rda_log_magnitude(radar) if include_rda else None
        phase_meta.update(
            {
                "method_pipeline": list(METHOD_PIPELINE),
                "innovation_modules": list(INNOVATION_MODULES),
                "domain_adaptation_method": DOMAIN_ADAPTATION_METHOD,
                "representation_method": "EDACM phase representation + HR-AdaVMD decomposition + FFT spectrum for CycleFormer",
                "representation_alias": representation,
                "vmd_k": DEFAULT_VMD_K,
                "vmd_alpha": DEFAULT_VMD_ALPHA,
                "vmd_max_iter": DEFAULT_VMD_MAX_ITER,
                "vmd_tol": DEFAULT_VMD_TOL,
                "vmd_output_shape": list(x_time.shape),
                **hr_adavmd_meta,
                "feature_domains": ["time", "frequency"],
                "x_time_shape": list(x_time.shape),
                "x_freq_shape": list(x_freq.shape),
                "x_rda_shape": list(x_rda.shape) if x_rda is not None else None,
                "x_rda_representation": DEFAULT_RDA_REPRESENTATION if x_rda is not None else None,
                **freq_meta,
            }
        )
        bundle = {
            "x": x_time.astype(np.float32),
            "x_time": x_time.astype(np.float32),
            "x_freq": x_freq.astype(np.float32),
            "freq_hz": freq_hz.astype(np.float32),
            "meta": phase_meta,
        }
        if x_rda is not None:
            bundle["x_rda"] = x_rda.astype(np.float32)
        return bundle
    else:
        raise ValueError(f"Unsupported representation: {representation}")
    return {"x": x, "meta": {}}
