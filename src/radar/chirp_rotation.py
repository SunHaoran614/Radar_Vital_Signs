"""Eight-group phase progression diagnostics, without velocity or HR calibration."""
import numpy as np

def chirp_rotation(z):
    z=np.asarray(z,dtype=np.complex128)
    if z.ndim<2 or z.shape[-1]!=8 or not np.isfinite(z).all():raise ValueError('Finite complex data with eight ordered groups required')
    scale=np.max(np.abs(z),axis=-1);u=np.divide(z,scale[...,None],out=np.zeros_like(z),where=scale[...,None]>0)
    power=np.sum(abs(u)**2,axis=-1);r=[];q=[]
    for k in range(1,8):
        front=u[...,:-k];back=u[...,k:];rk=np.sum(back*np.conj(front),axis=-1)
        denominator=np.sqrt(np.sum(abs(front)**2,axis=-1)*np.sum(abs(back)**2,axis=-1))
        r.append(rk);q.append(np.divide(rk,denominator,out=np.zeros_like(rk),where=denominator>0))
    r=np.stack(r,axis=-1);q=np.stack(q,axis=-1);valid=abs(r)>0
    theta=np.where(valid[...,0],np.angle(r[...,0]),np.nan);safe=np.where(valid[...,0],theta,0)
    error=np.where(valid & valid[...,:1],np.angle(r*np.exp(-1j*safe[...,None]*np.arange(1,8))),np.nan)
    basis=np.exp(1j*safe[...,None]*np.arange(8));amplitude=np.mean(u*np.conj(basis),axis=-1)
    constant=np.sum(abs(u-u.mean(axis=-1,keepdims=True))**2,axis=-1)
    residual=np.sum(abs(u-amplitude[...,None]*basis)**2,axis=-1)
    cf=np.divide(constant,power,out=np.full_like(power,np.nan),where=power>0)
    rf=np.divide(residual,power,out=np.full_like(power,np.nan),where=valid[...,0])
    ratio=np.divide(residual,constant,out=np.full_like(power,np.nan),where=valid[...,0] & (constant>0))
    early=np.sum(u[...,1:4]*np.conj(u[...,:3]),axis=-1);late=np.sum(u[...,5:8]*np.conj(u[...,4:7]),axis=-1)
    half=np.where((abs(early)>0)&(abs(late)>0),np.angle(late*np.conj(early)),np.nan)
    return dict(theta=theta,q1=q[...,0],coherence=abs(q),lag_error=error,constant_fraction=cf,
                rotation_fraction=rf,rotation_to_constant=ratio,half_gap=half)
