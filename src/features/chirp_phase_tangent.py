"""Local phase directions of the frozen bounded-correlation representation.

These directions rotate all returns, not a validated cardiac motion simulator.
"""
import numpy as np


def phase_weights(mode):
    if mode=='common':return np.ones(8)
    if mode=='slope':
        b=np.arange(8,dtype=float)-3.5
        return b/np.sqrt(np.mean(b*b))
    raise ValueError(mode)


def tangent(groups,signal,weights):
    """Analytic d features / d amplitude for z*exp(i*a*s[t]*b[g])."""
    x=np.asarray(groups,np.complex128);s=np.asarray(signal,float);b=np.asarray(weights,float)
    if x.ndim!=3 or x.shape[:2]!=(256,8) or s.shape!=(256,) or b.shape!=(8,):raise ValueError('shape')
    if not all(np.isfinite(v).all() for v in (x,s,b)):raise ValueError('nonfinite')
    scale=abs(x).max((0,1),keepdims=True);z=np.divide(x,scale,out=np.zeros_like(x),where=scale>0)
    dz=1j*s[:,None,None]*b[None,:,None]*z
    m=z.mean(1);dm=dz.mean(1);a=m[1:];c=m[:-1];da=dm[1:];dc=dm[:-1]
    den=abs(a)**2+abs(c)**2;dd=2*(np.real(da*a.conj())+np.real(dc*c.conj()))
    num=2*a*c.conj();dn=2*(da*c.conj()+a*dc.conj())
    # A phase direction can leave an exactly cancelled coherent mean.
    singular=(den==0)&((abs(da)>0)|(abs(dc)>0))
    if np.any(singular):raise ValueError('non-differentiable coherent-mean zero')
    q=np.divide(num,den,out=np.zeros_like(num),where=den>0)
    dq=np.divide(dn-q*dd,den,out=np.zeros_like(num),where=den>0)
    out=np.zeros((x.shape[2],16,256));out[:,0,1:]=dq.real.T;out[:,1,1:]=dq.imag.T
    for k in range(1,8):
        a=z[:,k:];c=z[:,:-k]
        dn=1j*s[:,None]*np.sum((b[k:]-b[:-k])[None,:,None]*a*c.conj(),axis=1)
        den=np.sqrt(np.sum(abs(a)**2,axis=1)*np.sum(abs(c)**2,axis=1))
        dq=np.divide(dn,den,out=np.zeros_like(dn),where=den>0)
        out[:,2*k]=dq.real.T;out[:,2*k+1]=dq.imag.T
    return out


def independent_tangent(groups,signal,weights):
    """Independent real/imag product and quotient derivatives, fixed scale."""
    x=np.asarray(groups,np.complex128);v=np.hypot(x.real,x.imag).max((0,1),keepdims=True)
    r=np.divide(x.real,v,out=np.zeros_like(x.real),where=v>0);i=np.divide(x.imag,v,out=np.zeros_like(x.imag),where=v>0)
    p=np.asarray(signal)[:,None,None]*np.asarray(weights)[None,:,None];dr=-p*i;di=p*r
    result=np.zeros((x.shape[2],16,256))
    def ratio(ar,ai,br,bi,dar,dai,dbr,dbi,group):
        nr=ar*br+ai*bi;ni=ai*br-ar*bi
        dnr=dar*br+ar*dbr+dai*bi+ai*dbi;dni=dai*br+ai*dbr-dar*bi-ar*dbi
        if group:
            nr=nr.sum(1);ni=ni.sum(1);dnr=dnr.sum(1);dni=dni.sum(1)
            aa=(ar*ar+ai*ai).sum(1);bb=(br*br+bi*bi).sum(1)
            daa=2*(ar*dar+ai*dai).sum(1);dbb=2*(br*dbr+bi*dbi).sum(1)
            den=np.sqrt(aa)*np.sqrt(bb)
            dd=.5*den*(np.divide(daa,aa,out=np.zeros_like(aa),where=aa>0)+np.divide(dbb,bb,out=np.zeros_like(bb),where=bb>0))
        else:
            nr*=2;ni*=2;dnr*=2;dni*=2
            den=ar*ar+ai*ai+br*br+bi*bi;dd=2*(ar*dar+ai*dai+br*dbr+bi*dbi)
            if np.any((den==0)&((dar!=0)|(dai!=0)|(dbr!=0)|(dbi!=0))):raise ValueError('non-differentiable coherent-mean zero')
        qr=np.divide(nr,den,out=np.zeros_like(nr),where=den>0);qi=np.divide(ni,den,out=np.zeros_like(ni),where=den>0)
        return np.divide(dnr-qr*dd,den,out=np.zeros_like(nr),where=den>0),np.divide(dni-qi*dd,den,out=np.zeros_like(ni),where=den>0)
    mr=r.sum(1)/8;mi=i.sum(1)/8;dmr=dr.sum(1)/8;dmi=di.sum(1)/8
    rr,ii=ratio(mr[1:],mi[1:],mr[:-1],mi[:-1],dmr[1:],dmi[1:],dmr[:-1],dmi[:-1],False)
    result[:,0,1:]=rr.T;result[:,1,1:]=ii.T
    for k in range(1,8):
        rr,ii=ratio(r[:,k:],i[:,k:],r[:,:-k],i[:,:-k],dr[:,k:],di[:,k:],dr[:,:-k],di[:,:-k],True)
        result[:,2*k]=rr.T;result[:,2*k+1]=ii.T
    return result
