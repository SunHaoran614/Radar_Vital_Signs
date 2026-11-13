"""Sinusoidal phase regression with independent intercepts on observed runs.

Zero IQ has no phase. Runs never bridge missing original frame IDs. Numerical
gain measures this regression's fit, not cardiac identity or signal quality.
"""
from __future__ import annotations
import numpy as np


def segment_phase_gls(iq, time_s, frequencies):
    z=np.asarray(iq);t=np.asarray(time_s,dtype=np.float64);f=np.asarray(frequencies,dtype=np.float64)
    if z.ndim!=1 or not np.iscomplexobj(z) or not np.isfinite(z).all() or len(z)<2:
        raise ValueError('finite 1D complex IQ with at least two original frames required')
    if t.shape!=z.shape or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('finite strictly increasing original timestamps required')
    if f.ndim!=1 or not len(f) or not np.isfinite(f).all() or np.any(f<=0) or np.any(np.diff(f)<=0):
        raise ValueError('finite positive increasing frequencies required')
    mask=z!=0
    starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
    ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])+1
    runs=list(zip(starts.tolist(),ends.tolist()))
    n=int(mask.sum());dof=n-len(runs)
    y=np.zeros(len(z));design=np.stack((np.cos(2*np.pi*f[:,None]*(t-t[0])),np.sin(2*np.pi*f[:,None]*(t-t[0]))),axis=-1)
    for a,b in runs:
        phase=np.unwrap(np.angle(z[a:b]));y[a:b]=phase-phase.mean()
        design[:,a:b]-=design[:,a:b].mean(axis=1,keepdims=True)
    design[:,~mask]=0
    energy=float(y@y);power=np.zeros(len(f));rank=np.zeros(len(f),bool)
    gram=np.einsum('fnk,fnl->fkl',design,design)
    eigen=np.linalg.eigvalsh(gram)
    rank=(eigen[:,0]>1e-10*eigen[:,1]) & (dof>=3)
    status='accepted'
    if dof<3:status='insufficient_support'
    elif energy<=1e-20:status='constant_phase'
    elif not rank.all():status='rank_deficient'
    if dof>=3 and energy>1e-20:
        rhs=np.einsum('fnk,n->fk',design,y)
        safe=np.where(rank[:,None,None],gram,np.eye(2))
        beta=np.linalg.solve(safe,rhs[...,None])[...,0]
        residual=y[None,:]-np.einsum('fnk,fk->fn',design,beta)
        gain=1-np.sum(residual*residual,axis=1)/energy
        assert np.all(gain[rank]>=-1e-10) and np.all(gain[rank]<=1+1e-10), 'gain outside numerical bounds'
        power=np.where(rank,np.clip(gain,0,1),0.)
    peak=int(np.argmax(power));maximum=float(power[peak]);ties=int(np.sum(abs(power-maximum)<=1e-10*(1+abs(maximum))))
    if status=='accepted':
        if maximum<=1e-10:status='no_evidence'
        elif ties!=1:status='ambiguous_peak'
    return dict(version='segment_phase_gls_v1',status=status,accepted=status=='accepted',
                candidate_bpm=float(60*f[peak]) if status=='accepted' else None,
                observed_count=n,segment_count=len(runs),effective_dof=dof,runs=runs,
                observed_mask=mask,power=power,rank_valid=rank,centered_phase=y,
                centered_energy=energy,tied_peak_grid_points=ties,cardiac_validated=False)
