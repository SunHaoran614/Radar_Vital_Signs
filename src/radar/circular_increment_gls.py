"""Versioned complex adjacent-increment GLS with explicit unobserved samples and abstention.

Exact-zero IQ has no phase. Dropping it must not compress the time grid.
This is a numerical estimator, not certification of cardiac signal identity.
"""
from __future__ import annotations
import numpy as np

STATUS = {0: 'valid', 1: 'insufficient_pairs', 2: 'constant_increment',
          3: 'rank_deficient', 4: 'no_evidence', 5: 'ambiguous_peak'}
RANK_RTOL=1e-10
PEAK_TOL=1e-10
VAR_FLOOR=1e-20

def peak_status(power, counts, variance, rank_valid):
    """Apply the fixed numerical identifiability/readout contract."""
    p=np.asarray(power,np.float64);counts=np.asarray(counts);variance=np.asarray(variance);rank_valid=np.asarray(rank_valid,bool)
    maximum=p.max(-1);ties=(abs(p-maximum[...,None])<=PEAK_TOL*(1+abs(maximum[...,None]))).sum(-1)
    status=np.zeros(counts.shape,np.int8)
    status=np.where(ties!=1,5,status);status=np.where(maximum<=PEAK_TOL,4,status)
    status=np.where(~rank_valid.all(-1),3,status);status=np.where(variance<=VAR_FLOOR,2,status);status=np.where(counts<4,1,status)
    return status.astype(np.int8)

def circular_increment_gls(cells, time_s, frequencies):
    """Fit a floating-mean sinusoid at each frequency on observed increments.

    Return power in [0,1], pair count, per-frequency design rank validity,
    status code, variance, and candidate BPM (NaN when not identifiable).
    ``time_s`` refers to every original frame, including exact-zero IQ frames.
    """
    cells=np.asarray(cells)
    if not np.iscomplexobj(cells) or cells.ndim<1 or cells.shape[-1]<2 or not np.isfinite(cells).all():raise ValueError('finite complex input with last axis >=2 required')
    z=cells.astype(np.complex128);t=np.asarray(time_s,np.float64);freq=np.asarray(frequencies,np.float64)
    if t.ndim!=1 or len(t)!=z.shape[-1] or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('strictly increasing finite original time grid required')
    if freq.ndim!=1 or not len(freq) or not np.isfinite(freq).all() or np.any(freq<=0) or np.any(np.diff(freq)<=0):raise ValueError('positive increasing finite frequencies required')
    shape=z.shape[:-1];flat=z.reshape(-1,z.shape[-1]);observed=(flat[:,1:]!=0)&(flat[:,:-1]!=0)
    scale=np.maximum(abs(flat.real),abs(flat.imag));scaled=np.divide(flat,scale,out=np.zeros_like(flat),where=scale>0)
    norm=np.sqrt(scaled.real**2+scaled.imag**2);unit=np.divide(scaled,norm,out=np.zeros_like(flat),where=norm>0)
    increment=np.where(observed,unit[:,1:]*unit[:,:-1].conj(),0j)
    counts=observed.sum(-1);mean=increment.sum(-1)/np.maximum(counts,1);centered=np.where(observed,increment-mean[:,None],0j);energy=(abs(centered)**2).sum(-1);variance=energy/np.maximum(counts,1)
    midpoint=(t[1:]+t[:-1])/2;midpoint=midpoint-midpoint[0];angle=2*np.pi*freq[:,None]*midpoint[None,:];design=np.stack((np.cos(angle),np.sin(angle),np.ones_like(angle)),axis=-1)
    output=np.zeros((len(flat),len(freq)));ranks=np.zeros_like(output,bool)
    for start in range(0,len(flat),128):
        mask=observed[start:start+128].astype(np.float64);values=centered[start:start+128]
        gram=np.einsum('kn,fnj,fnl->kfjl',mask,design,design,optimize=True)
        rhs=np.einsum('kn,fnj->kfj',values,design,optimize=True)
        eigen=np.linalg.eigvalsh(gram);rank=(eigen[...,0]>RANK_RTOL*eigen[...,-1])&(counts[start:start+128,None]>=4)
        # Replace singular systems only for computation; their output is invalid.
        safe=np.where(rank[...,None,None],gram,np.eye(3));beta=np.linalg.solve(safe,rhs[...,None])[...,0]
        fitted=np.einsum('fnj,kfj->kfn',design,beta,optimize=True)
        residual=np.where(observed[start:start+128,None,:],values[:,None,:]-fitted,0.)
        sse=(abs(residual)**2).sum(-1);den=energy[start:start+128,None]
        gain=1-np.divide(sse,den,out=np.ones_like(sse),where=den>0)
        output[start:start+128]=np.where(rank&(variance[start:start+128,None]>VAR_FLOOR),np.clip(gain,0,1),0.);ranks[start:start+128]=rank
    status=peak_status(output,counts,variance,ranks);candidate=freq[output.argmax(-1)]*60;candidate=np.where(status==0,candidate,np.nan)
    return dict(power=output.reshape(shape+(len(freq),)),pair_count=counts.reshape(shape),rank_valid=ranks.reshape(shape+(len(freq),)),variance=variance.reshape(shape),status=status.reshape(shape),candidate_bpm=candidate.reshape(shape))
