"""Experimental angle-bin weighting stage for an already converted cube.

The angle axis can contain angle FFT bins or raw antenna channels, depending on
the loader. This module receives no provenance or array calibration. The
``bartlett`` and ``mvdr`` implementations below are angle-bin weighting proxies;
they must not be interpreted as physical beamformers or used to claim an MVDR
benefit. Real beamforming belongs before the angle FFT on calibrated RX IQ.

Candidate matrix rows: B0 (identity on input spatial axis), B1/B2 (uncalibrated
proxies requiring replacement before any physical beamforming claim).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .pipeline import register_stage


def _complex_safe(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.complex64)


def _angle_covariance(cube: np.ndarray) -> np.ndarray:
    """Covariance of the input angle bins, averaged over (F, D, R).

    Returns (A, A) complex matrix.
    """
    f, d, a, r = cube.shape
    flat = cube.transpose(1, 3, 0, 2).reshape(f * r * d, a)  # (cells, A)
    flat = flat - flat.mean(axis=0, keepdims=True)
    cov = (flat.conj().T @ flat) / max(1, flat.shape[0] - 1)
    return cov


def beamforming_fft(cube: np.ndarray, **_: Any) -> np.ndarray:
    """B0: identity on an existing cube; does not compute an angle FFT here."""
    return _complex_safe(cube)


def beamforming_bartlett(cube: np.ndarray, look_directions: int = 16, **_: Any) -> np.ndarray:
    """B1: uncalibrated angle-bin spectral weighting proxy.

    A synthetic steering vector reweights the input bins. If those bins already
    contain an angle FFT, this calculation has no calibrated RX-domain meaning.
    """
    cube = _complex_safe(cube)
    f, d, a, r = cube.shape
    # steering vectors: crude linear array assumption, angle bin -> direction cosine
    idx = np.arange(a, dtype=np.float32)
    # normalized spatial frequency per angle bin (treat bin index as direction proxy)
    u = np.linspace(-0.5, 0.5, a, dtype=np.float32)
    # Bartlett power per (f, d, r) cell across angle
    out = np.zeros_like(cube)
    flat = cube.transpose(1, 3, 0, 2).reshape(d * r * f, a)  # (cells, A)
    # steering matrix: A x A, S[k,j] = exp(-2j pi u[k] * j)
    j = np.arange(a, dtype=np.float32)
    S = np.exp(-2j * np.pi * np.outer(u, j)).astype(np.complex64)  # (A, A)
    # Bartlett spectrum: |S^H x|^2 ; gives (cells, A)
    spec = np.abs((S.conj().T @ flat.T)).T ** 2  # (cells, A)
    spec = spec.reshape(d, r, f, a).transpose(2, 0, 3, 1)  # (F, D, A, R)
    # normalize per (F,D,R) so the strongest steering direction is kept ~unity
    spec = spec / (spec.max(axis=2, keepdims=True) + 1e-8)
    return cube * spec


def beamforming_mvdr(cube: np.ndarray, diag_load: float = 1e-2, **_: Any) -> np.ndarray:
    """B2: uncalibrated covariance-based angle-bin weighting proxy.

    Covariance across input bins and synthetic steering vectors yield a weight.
    The input bins may already be angle FFT outputs, so this is not a physical
    RX-domain MVDR beamformer.
    """
    cube = _complex_safe(cube)
    f, d, a, r = cube.shape
    cov = _angle_covariance(cube)  # (A, A)
    # diagonal load for numerical stability on tiny cubes
    cov_loaded = cov + diag_load * np.trace(cov) / a * np.eye(a, dtype=np.complex64)
    try:
        inv = np.linalg.inv(cov_loaded)
    except np.linalg.LinAlgError:
        return cube
    # steering vectors per angle bin (same proxy as Bartlett)
    u = np.linspace(-0.5, 0.5, a, dtype=np.float32)
    j = np.arange(a, dtype=np.float32)
    S = np.exp(-2j * np.pi * np.outer(u, j)).astype(np.complex64)  # (A, A)
    # MVDR angular gain per steering direction k: 1 / (s_k^H R^-1 s_k)
    gains = np.zeros(a, dtype=np.float32)
    for k in range(a):
        s = S[k]
        denom = np.real(s.conj() @ inv @ s)
        gains[k] = 1.0 / max(float(denom), 1e-8)
    # normalize so max gain ~ unity
    gains = gains / (gains.max() + 1e-8)
    # apply gain per angle bin (same gain across all F,D,R for this cheap version)
    weight = gains[None, None, :, None].astype(np.complex64)
    return cube * weight


def register_beamforming_methods() -> None:
    register_stage("beamforming", "fft", beamforming_fft)
    register_stage("beamforming", "bartlett", beamforming_bartlett)
    register_stage("beamforming", "mvdr", beamforming_mvdr)
