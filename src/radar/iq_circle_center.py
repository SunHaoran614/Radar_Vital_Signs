"""Radar-only algebraic circle-center calibration; no whitening or label use."""
import numpy as np

def fit_record_centers(z):
    z=np.asarray(z,np.complex128)
    if z.ndim!=2 or not np.isfinite(z).all():raise ValueError('Expected finite time x cell complex data')
    mean=z.mean(0);delta=z-mean;scale=np.sqrt(np.mean(abs(delta)**2,axis=0));u=np.divide(delta,scale,out=np.zeros_like(delta),where=scale>0)
    center=np.zeros(z.shape[1],np.complex128);condition=np.zeros(z.shape[1]);fallback=np.ones(z.shape[1],bool)
    for k in range(z.shape[1]):
        design=np.column_stack((2*u[:,k].real,2*u[:,k].imag,np.ones(len(u))))
        beta,_,_,sv=np.linalg.lstsq(design,abs(u[:,k])**2,rcond=None);condition[k]=sv[-1]/sv[0]
        if scale[k]>0 and condition[k]>=1e-6:center[k]=complex(beta[0],beta[1]);fallback[k]=False
    radius=abs(u-center);radius_mean=radius.mean(0);cv=np.divide(radius.std(0),radius_mean,out=np.zeros_like(radius_mean),where=radius_mean>0)
    return dict(mean=mean,scale=scale,u=u,center=center,condition=condition,fallback=fallback,radius_cv=cv)
