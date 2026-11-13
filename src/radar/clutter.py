"""Clutter suppression stage: remove static / stationary reflection.

The cube is complex ``(F, D, A, R)``. Stationary reflectors (walls, seat frames,
body mass not moving) produce a slow-time (frame-axis) component with ~zero
Doppler; vital-sign micro-motion lives in the fast oscillatory part. All methods
here operate along the frame axis (axis 0) and return a cube of the same shape.

Candidate matrix rows (see docs/frontend.md): C0..C6.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .pipeline import register_stage, DEFAULT_FS


def _complex_safe(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.complex64)


def clutter_none(cube: np.ndarray, **_: Any) -> np.ndarray:
    """Baseline: preserve the input, including any stationary reflections."""
    return _complex_safe(cube)


def clutter_mean_subtraction(cube: np.ndarray, **_: Any) -> np.ndarray:
    """C0: subtract the slow-time mean per (D, A, R) cell (zeros out DC)."""
    cube = _complex_safe(cube)
    mean = cube.mean(axis=0, keepdims=True)
    return cube - mean


def clutter_temporal_highpass(cube: np.ndarray, cutoff: float = 0.1,
                              fs: float = DEFAULT_FS, **_: Any) -> np.ndarray:
    """C1: 1st-order IIR high-pass along frames to kill sub-``cutoff``-Hz drift.

    ``cutoff`` is in Hz relative to the frame rate; mapped to an alpha via the
    standard one-pole high-pass recurrence. Cheap, label-free, fully reversible
    shape-wise.
    """
    cube = _complex_safe(cube)
    if (not np.isfinite(cutoff) or cutoff <= 0 or not np.isfinite(fs)
            or fs <= 0 or cutoff >= fs / 2):
        raise ValueError("cutoff must lie in (0, fs/2), with finite positive fs")
    n = cube.shape[0]
    if n < 2:
        return cube
    # RC = 1/(2*pi*f_c); omitting 2*pi makes the effective cutoff ~6x too low.
    rc = 1.0 / (2.0 * np.pi * float(cutoff))
    dt = 1.0 / float(fs)
    alpha = rc / (rc + dt)
    out = np.empty_like(cube)
    out[0] = 0.0
    for t in range(1, n):
        out[t] = alpha * (out[t - 1] + cube[t] - cube[t - 1])
    return out


def clutter_mti(cube: np.ndarray, **_: Any) -> np.ndarray:
    """C2: 2-pulse MTI canceler (frame difference) to remove zero-Doppler clutter."""
    cube = _complex_safe(cube)
    if cube.shape[0] < 2:
        return np.zeros_like(cube)
    diff = np.diff(cube, axis=0)
    # No preceding frame exists at t=0: zero avoids a spurious impulse from
    # a stationary reflector while preserving the (F,D,A,R) shape contract.
    return np.concatenate([np.zeros_like(cube[:1]), diff], axis=0)


def clutter_pca(cube: np.ndarray, n_components: int = 1, **_: Any) -> np.ndarray:
    """C3: remove dominant *complex* shared slow-time modes across RDA cells.

    Per-cell temporal means are preserved. A complex rank-one motion with
    different spatial phases is one mode; stacking real and imaginary samples
    as separate time rows can split it into two artificial real modes.
    """
    cube = _complex_safe(cube)
    f, d, a, r = cube.shape
    flat = cube.reshape(f, d * a * r)  # (F, cells)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    try:
        temporal_modes, _, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return cube
    k = min(max(int(n_components), 0), temporal_modes.shape[1])
    leading = temporal_modes[:, :k]
    cleaned = (centered - leading @ (leading.conj().T @ centered)
               if k > 0 else centered) + mean
    return cleaned.reshape(f, d, a, r).astype(np.complex64)


def clutter_rpca(cube: np.ndarray, lamb: float = 0.05, n_iter: int = 30, **_: Any) -> np.ndarray:
    """C4: lightweight low-rank + sparse (RPCA) split via inexact ALM on slow-time.

    Models the cube as low-rank (static / structured clutter) + sparse (micro-
    motion transients). We keep the *sparse* part as the cleaned signal. This is a
    reduced-complexity PCP, not a full RPCA, intended for CPU sanity checks.
    """
    cube = _complex_safe(cube)
    f, d, a, r = cube.shape
    mat = cube.reshape(f, d * a * r).astype(np.complex64)
    # operate on magnitude-normalised real matrix for stability
    mag = np.abs(mat)
    norm = float(np.max(mag)) + 1e-8
    X = (mat / norm).astype(np.complex64)
    # real + imag stacked
    Xr = np.concatenate([X.real, X.imag], axis=0)
    m, n = Xr.shape
    Y = np.zeros_like(Xr)
    mu = 1.0 / (4.0 * np.linalg.norm(Xr, 2) + 1e-8)
    svthr = 1.0 / mu
    lrank = np.zeros_like(Xr)
    for _ in range(n_iter):
        # sparse update (hard threshold)
        S = np.sign(Xr - lrank + Y / mu) * np.maximum(
            np.abs(Xr - lrank + Y / mu) - lamb / mu, 0.0
        )
        # low-rank update (singular value threshold)
        U, s, Vt = np.linalg.svd(Xr - S + Y / mu, full_matrices=False)
        s_thr = np.maximum(s - svthr, 0.0)
        lrank = (U * s_thr) @ Vt
        Y = Y + mu * (Xr - lrank - S)
    sparse = (Xr - lrank)[:f] + 1j * (Xr - lrank)[f:]
    return (sparse * norm).reshape(f, d, a, r).astype(np.complex64)


def clutter_complex_rpca(cube: np.ndarray, lamb: float | None = None,
                         n_iter: int = 100, tol: float = 1e-5,
                         **_: Any) -> np.ndarray:
    """Complex principal-component pursuit, returning its sparse component.

    Unlike ``rpca`` this keeps I/Q as one complex entry and uses complex
    magnitude soft thresholding. The low-rank/sparse assumption is a *candidate*
    clutter model: an oscillating heart scatterer need not be sparse enough for
    PCP to preserve it. Callers must test physical and HR information retention.

    Solve in units of maximum input magnitude, then restore the original gain.
    PCP is homogeneous: changing receiver gain must not change the decomposition
    merely because an absolute numerical floor exceeds the input amplitude.
    """
    cube = _complex_safe(cube)
    if cube.ndim != 4 or min(cube.shape) < 1:
        raise ValueError('Expected a nonempty (frames,doppler,angle,range) cube')
    if not np.isfinite(cube).all() or n_iter < 1 or not 0 < tol < 1:
        raise ValueError('Invalid complex RPCA input or iteration controls')
    matrix = cube.reshape(cube.shape[0], -1).astype(np.complex128)
    rows, cols = matrix.shape
    weight = 1.0 / np.sqrt(max(rows, cols)) if lamb is None else float(lamb)
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('lamb must be finite and positive')
    scale = float(np.max(abs(matrix)))
    if scale == 0:
        return np.zeros_like(cube)
    matrix /= scale
    norm_fro = float(np.linalg.norm(matrix))
    norm_two = float(np.linalg.norm(matrix, ord=2))
    dual = matrix / max(norm_two, float(np.max(abs(matrix))) / weight)
    mu = 1.25 / norm_two
    mu_limit = mu * 1e7
    low_rank = np.zeros_like(matrix)
    sparse = np.zeros_like(matrix)
    for _ in range(n_iter):
        previous_low_rank, previous_sparse = low_rank, sparse
        u, singular, vh = np.linalg.svd(matrix - sparse + dual / mu, full_matrices=False)
        retained = np.maximum(singular - 1.0 / mu, 0.0)
        low_rank = (u * retained) @ vh
        residual = matrix - low_rank + dual / mu
        magnitude = abs(residual)
        shrink = np.divide(np.maximum(magnitude - weight / mu, 0.0),
                           magnitude, out=np.zeros_like(magnitude),
                           where=magnitude > 0)
        sparse = residual * shrink
        error = matrix - low_rank - sparse
        # Feasibility alone can hold on the very first iteration while the
        # nuclear/l1 objective is still decreasing (e.g. one isolated entry).
        change = max(np.linalg.norm(low_rank - previous_low_rank),
                     np.linalg.norm(sparse - previous_sparse)) / norm_fro
        if np.linalg.norm(error) / norm_fro <= tol and change <= tol:
            break
        dual += mu * error
        mu = min(mu * 1.5, mu_limit)
    return (sparse * scale).reshape(cube.shape).astype(np.complex64)


def register_clutter_methods() -> None:
    register_stage("clutter", "none", clutter_none)
    register_stage("clutter", "mean_subtraction", clutter_mean_subtraction)
    register_stage("clutter", "temporal_highpass", clutter_temporal_highpass)
    register_stage("clutter", "mti", clutter_mti)
    register_stage("clutter", "pca", clutter_pca)
    register_stage("clutter", "rpca", clutter_rpca)
    register_stage("clutter", "complex_rpca", clutter_complex_rpca)
