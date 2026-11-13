"""Complex RX-domain beamforming with explicit steering vectors.

Inputs are snapshots x RX, never angle-FFT bins. The caller owns channel order,
calibration and steering provenance. No dataset geometry is inferred here.
Convention: x(t)=a*s(t), y(t)=w^H*x(t), and w^H*a=1.
MVDR formula: https://www.mathworks.com/help/phased/ref/phased.mvdrbeamformer-system-object.html
"""
from __future__ import annotations

import numpy as np


def _matrix(values, name):
    values = np.asarray(values, dtype=np.complex128)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError(f'{name} must be a nonempty finite 2D matrix')
    return values


def bartlett_weights(steering):
    """Return beams x RX weights, normalized to unit complex target gain."""
    a = _matrix(steering, 'steering')
    energy = np.sum(abs(a)**2, axis=1)
    if np.any(energy <= 0):
        raise ValueError('Zero steering vector')
    return a/energy[:, None]


def mvdr_weights(snapshots, steering, *, diagonal_loading=0.01):
    """Use uncentered RX covariance and trace-relative diagonal loading.

    Weights are fixed across the supplied snapshots. The caller must specify
    whether those snapshots include future samples or the target signal.
    Singular zero-power covariance is rejected rather than silently bypassed.
    """
    x = _matrix(snapshots, 'snapshots')
    a = _matrix(steering, 'steering')
    if x.shape[1] != a.shape[1]:
        raise ValueError('RX count mismatch')
    if not np.isfinite(diagonal_loading) or diagonal_loading < 0:
        raise ValueError('diagonal_loading must be finite and nonnegative')
    bartlett_weights(a)  # also rejects zero steering vectors
    covariance = x.T @ x.conj()/len(x)
    scale = float(np.trace(covariance).real/x.shape[1])
    if scale <= 0:
        raise ValueError('Zero-power covariance')
    covariance += diagonal_loading*scale*np.eye(x.shape[1])
    solved = np.linalg.solve(covariance, a.T).T
    denominators = np.sum(a.conj()*solved, axis=1)
    if np.any(denominators.real <= 0) or not np.isfinite(solved).all():
        raise ValueError('Invalid covariance solution')
    return solved/denominators[:, None]


def apply_weights(snapshots, weights):
    """Return snapshots x beams complex IQ; never a power-only spectrum."""
    x = _matrix(snapshots, 'snapshots')
    w = _matrix(weights, 'weights')
    if x.shape[1] != w.shape[1]:
        raise ValueError('RX count mismatch')
    return x @ w.conj().T
