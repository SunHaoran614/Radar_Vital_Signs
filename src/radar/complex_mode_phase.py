"""Full complex PCA before nonlinear phase extraction; no rank truncation."""
import numpy as np
from src.radar.iq_circle_center import fit_record_centers

def basis(z,sham=False):
    z=np.asarray(z,np.complex128)
    if z.ndim!=2 or z.shape[1]!=48 or not np.isfinite(z).all():raise ValueError('Expected finite time x 48')
    centered=z-z.mean(0)
    estimate=centered.copy()
    if sham:
        for c in range(48):estimate[:,c]=np.roll(centered[:,c],c*len(z)//48)
    _,s,vh=np.linalg.svd(estimate,full_matrices=False)
    return centered,vh.conj().T,s

def modes(z,sham=False):
    centered,w,s=basis(z,sham)
    original=centered@w
    rms=np.sqrt(np.mean(abs(original)**2,axis=0));null=rms<=64*np.finfo(float).eps*max(float(rms.max()),1e-300)
    clean=original.copy();clean[:,null]=0
    fit=fit_record_centers(clean)
    return dict(centered=centered,basis=w,singular=s,raw_modes=original,clean_modes=clean,null=null,fit=fit,mean=fit['u'],circle=fit['u']-fit['center'])
