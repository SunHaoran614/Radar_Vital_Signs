"""Fixed orthogonal coordinates for eight ordered coherent chirp groups.

Coordinates retain the common mean, the original first/last contrast, and its
six-dimensional orthogonal complement. They do not identify cardiac motion.
"""
import numpy as np


def basis():
    """Columns: unit common mode, unit first32-last32, six fixed complements."""
    columns=[np.ones(8)/np.sqrt(8),np.array([1,1,0,0,0,0,-1,-1],float)/2]
    for k in range(1,8):
        value=np.cos(np.pi*(np.arange(8)+.5)*k/8)
        # Two-pass Gram-Schmidt fixes coordinates without data-dependent fitting.
        for _ in range(2):
            for col in columns:value=value-col*np.dot(col,value)
        length=np.linalg.norm(value)
        if length>1e-10:
            value=value/length
            if value[np.flatnonzero(abs(value)>1e-10)[0]]<0:value=-value
            columns.append(value)
        if len(columns)==8:break
    q=np.column_stack(columns)
    assert q.shape==(8,8) and np.max(abs(q.T@q-np.eye(8)))<1e-12
    return q


def encode(value):
    """Return normalized complex coordinates and per-cell record energy.

    Input [frame,8,cell]. Record-wide scaling uses future radar and no labels.
    The energy is retained so normalization is reversible above the floor.
    """
    x=np.asarray(value,dtype=np.complex128)
    if x.ndim!=3 or x.shape[1]!=8 or min(x.shape)==0 or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty complex [frame,8,cell]')
    energy=np.mean(abs(x)**2,axis=(0,1));scale=np.sqrt(np.maximum(energy,1e-12))
    coefficients=np.einsum('gk,tgc->tkc',basis(),x)/scale[None,None,:]
    return coefficients,energy


def decode(coefficients,energy):
    c=np.asarray(coefficients,dtype=np.complex128);e=np.asarray(energy,dtype=float)
    if c.ndim!=3 or c.shape[1]!=8 or e.shape!=(c.shape[-1],) or np.any(e<0):
        raise ValueError('Incompatible coordinates and energy')
    return np.einsum('gk,tkc->tgc',basis(),c)*np.sqrt(np.maximum(e,1e-12))[None,None,:]
