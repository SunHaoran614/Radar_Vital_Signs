"""Label-free spatial nuisance projection for time-by-cell radar signals.

This assumes nuisance and cardiac motion have different spatial signatures.
It deliberately cannot protect a cardiac component in the nuisance subspace.
"""
from __future__ import annotations
import numpy as np
from scipy import signal


def standardize_cells(x: np.ndarray) -> np.ndarray:
    x=signal.detrend(np.asarray(x),axis=0)
    return x/np.maximum(np.sqrt(np.mean(abs(x)**2,axis=0,keepdims=True)),1e-12)


def remove_spatial_subspace(x: np.ndarray, band: tuple[float,float],
                            rank: int=2, fs: float=10.) -> np.ndarray:
    """Estimate spatial covariance in a fixed band, then remove it at all times.

    Real input uses a real covariance; complex input retains complex weights.
    No Hann taper is used for covariance; downstream spectra apply their taper.
    """
    x=np.asarray(x)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Expected finite time-by-cell input')
    ff=np.fft.fftfreq(len(x),1/fs)
    keep=(abs(ff)>=band[0])&(abs(ff)<=band[1])
    if not keep.any() or not 1<=rank<x.shape[1]:raise ValueError('Invalid band or spatial rank')
    f=np.fft.fft(x,axis=0)[keep]
    covariance=f.conj().T@f
    if not np.iscomplexobj(x):covariance=covariance.real
    values,vectors=np.linalg.eigh(covariance)
    # Do not project arbitrary eigenvectors when the estimated nuisance is zero.
    threshold=max(float(values[-1])*1e-10,1e-20)
    indices=np.flatnonzero(values>threshold)[-rank:]
    v=vectors[:,indices]
    return x-(x@v)@v.conj().T
