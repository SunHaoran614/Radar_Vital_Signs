"""Phase-extraction candidates for the RDA front-end search pool.

Boundary note (design: "可替换的 RDA 前端算法池", §7):
    This module lives at the RDA -> representation seam. It supplies
    *alternative phase-extraction candidates* so the search pool can test
    whether the current EDACM path (owned by ``src/data/build_training_dataset.py``
    via ``edacm_phase`` / ``target_edacm_signal``) is excluding a simpler, more
    stable phase representation.

    These functions do NOT re-implement EDACM. They only produce candidate
    phase traces from a complex RDA cube given a pre-selected target bin
    (or a trivial global-energy selection when none is provided). If a
    candidate proves better than EDACM downstream, it is *promoted* into the
    representation layer, not duplicated here.

Shape contract: input is a complex cube ``(frames, doppler, angle, range)``.
Every function returns a 1-D float phase trace ``(frames,)``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np

from .numerics import magnitude_squared

# --- shared helpers -------------------------------------------------------


def _select_target_bin(cube: np.ndarray) -> Tuple[int, int, int]:
    """Trivial global-energy target-bin pick (kept inside phase.py on purpose).

    This is a *placeholder* localization so phase candidates can run without an
    upstream localization stage. The real pool uses the cube already localized
    by ``src/radar/localization.py``; callers that have a target bin should pass
    it via the helper wrappers below.
    """
    energy = np.mean(magnitude_squared(cube), axis=0)
    d_idx, a_idx, r_idx = np.unravel_index(int(np.argmax(energy)), energy.shape)
    return int(d_idx), int(a_idx), int(r_idx)


def _unwrap_phase(phase: np.ndarray) -> np.ndarray:
    """Unwrap a trace while retaining the same offset at every sample.

    The former loop retained the initial phase only at sample zero and made
    later samples relative to it, injecting an artificial first-sample impulse.
    """
    y = np.asarray(phase, dtype=np.float64)
    return np.unwrap(y).astype(np.float32)


def _zscore(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    if y.size == 0 or not np.isfinite(y).any():
        return np.zeros_like(y, dtype=np.float32)
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    return ((y - mean) / std).astype(np.float32) if (np.isfinite(std) and std > 1e-6) else (y - mean).astype(np.float32)


# --- candidates -----------------------------------------------------------


def phase_single_bin(
    cube: np.ndarray,
    target_bin: Tuple[int, int, int] | None = None,
    normalize: bool = True,
    **_: Any,
) -> np.ndarray:
    """P0: arctan phase of a single target bin, unwrapped + detrended.

    Cheapest possible phase: take one complex bin, unwrap its angle, remove a
    linear trend. No inter-bin fusion. Baseline "is raw phase usable at all?".

    Args:
        normalize: if True (default, for NN downstream) z-score the trace so its
            variance is ~1; if False return the raw detrended phase in radians,
            which is what the Phase 0 diagnostic needs (raw variance/continuity
            are meaningful only on the un-normalized signal).
    """
    if target_bin is None:
        target_bin = _select_target_bin(cube)
    d, a, r = target_bin
    z = cube[:, int(d), int(a), int(r)].astype(np.complex64)
    phase = np.angle(z).astype(np.float32)
    phase = _unwrap_phase(phase)
    t = np.linspace(-1.0, 1.0, phase.size, dtype=np.float32)
    valid = np.isfinite(phase)
    if int(valid.sum()) >= 2:
        slope, intercept = np.polyfit(t[valid], phase[valid], deg=1)
        phase = phase - (slope * t + intercept).astype(np.float32)
    return _zscore(phase) if normalize else phase


def phase_interframe_diff(
    cube: np.ndarray,
    target_bin: Tuple[int, int, int] | None = None,
    normalize: bool = True,
    **_: Any,
) -> np.ndarray:
    """P1: inter-frame phase difference (conjugate multiplication).

    The conjugate-multiplication trick: phase[t] = angle(z[t] * conj(z[t-1])).
    Numerically stable, immune to absolute phase offset, the workhorse behind
    most mmWave vital-sign phase pipelines (and a close cousin of EDACM's
    delta-phase). Compared against EDACM to see if the extra target-selection /
    fusion in EDACM earns its complexity.

    Args:
        normalize: see :func:`phase_single_bin`. Diagnostic callers pass False.
    """
    if target_bin is None:
        target_bin = _select_target_bin(cube)
    d, a, r = target_bin
    # Promote before the product: finite complex64 IQ may square outside the
    # float32 range even though its phase is well defined.
    z = cube[:, int(d), int(a), int(r)].astype(np.complex128)
    conj_prev = np.conj(z[:-1])
    prod = z[1:] * conj_prev
    delta = np.angle(prod).astype(np.float32)
    delta = _unwrap_phase(delta)
    # pad to original length, center
    full = np.concatenate([np.array([0.0], dtype=np.float32), delta])
    return _zscore(full) if normalize else full


def phase_multibin_fusion(
    cube: np.ndarray,
    top_bins: int = 3,
    target_bins: list[Tuple[int, int, int]] | None = None,
    normalize: bool = True,
    **_: Any,
) -> np.ndarray:
    """P2: weighted fusion of P1 traces over several high-energy bins.

    Runs ``phase_interframe_diff`` on the ``top_bins`` highest-energy bins and
    averages them with equal weight (a deliberately simple fusion; the
    representation layer's EDACM uses stability/HR-weighted fusion). Tests
    whether *more bins* helps or just adds noise. Kept simple here on purpose
    so the controlled variable is "how many bins", not "which fancy fusion".

    Args:
        normalize: see :func:`phase_single_bin`. Diagnostic callers pass False.
    """
    if target_bins is None:
        energy = np.mean(magnitude_squared(cube), axis=0)
        flat = np.argpartition(energy.reshape(-1), -top_bins)[-top_bins:]
        # np.unravel_index returns one array per axis; zip it to get per-bin
        # (d, a, r) tuples. Iterating it directly yields axis-index arrays and
        # the bin tuple unpack below would read garbage coordinates.
        target_bins = [tuple(int(i) for i in idx)
                       for idx in zip(*np.unravel_index(flat, energy.shape))]
    traces = [phase_interframe_diff(cube, tb, normalize=normalize) for tb in target_bins]
    stacked = np.stack(traces, axis=0).astype(np.float32)
    fused = np.mean(stacked, axis=0)
    return _zscore(fused) if normalize else fused


# --- registry -------------------------------------------------------------


PHASE_METHODS: Dict[str, Callable[..., np.ndarray]] = {
    "p0_single": phase_single_bin,
    "p1_diff": phase_interframe_diff,
    "p2_fusion": phase_multibin_fusion,
}


def register_phase_methods() -> Dict[str, Callable[..., np.ndarray]]:
    """Register phase candidates into the RDA stage registry.

    Called from ``src/radar/__init__.py`` at import time. The phase stage is
    cross-layer (it returns a 1-D trace, not a cube), but it is still part of the
    selectable front-end pool per docs/frontend.md §2.
    """
    from .pipeline import register_stage

    for name, func in PHASE_METHODS.items():
        register_stage("phase", name, func)
    return dict(PHASE_METHODS)
