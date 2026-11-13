"""Full-dimensional Hermitian shrink whitening; no labels or rank selection."""
import numpy as np

RHO=.1

def estimate(x,kind='real'):
    x=np.asarray(x,np.complex128)
    if x.ndim!=2 or x.shape[0]!=48 or not np.isfinite(x).all():raise ValueError('Expected finite 48 x time')
    if kind not in ('real','sham','diag'):raise ValueError(kind)
    z=x.copy()
    if kind=='sham':
        for c in range(48):z[c]=np.roll(z[c],c*z.shape[1]//48)
    c=z@z.conj().T/z.shape[1];c=(c+c.conj().T)/2
    if kind=='diag':c=np.diag(c.diagonal().real).astype(complex)
    return c

def operator(c):
    c=np.asarray(c,np.complex128);power=float(c.trace().real/48)
    if power==0:return np.eye(48,dtype=complex),np.eye(48,dtype=complex)
    a=(1-RHO)*c+RHO*power*np.eye(48);eig,u=np.linalg.eigh(a)
    if not np.all(eig>0):raise ValueError('Non-positive regularized covariance')
    h=(u/np.sqrt(eig)[None])@u.conj().T
    return h,a

def transform(x,kind='real'):
    c=estimate(x,kind);h,a=operator(c);y=h@x;scale=max(float(np.sqrt(np.mean(abs(y)**2))),1e-8)
    return dict(covariance=c,regularized=a,operator=h,scale=scale,output=y/scale)
