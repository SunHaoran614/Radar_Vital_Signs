"""Bounded group correlations with no per-channel amplitude normalization."""
import numpy as np
import torch
from torch import nn
from src.models.dar_cuda_equivalent import circular_shift_convolution


def window_features(groups):
    """[256,8,cell] complex -> [cell,16,256]; absent pairs map to zero."""
    x=np.asarray(groups,np.complex128)
    if x.ndim!=3 or x.shape[:2]!=(256,8) or not np.isfinite(x).all():
        raise ValueError('Expected finite [256,8,cell]')
    out=np.zeros((x.shape[2],16,256),float)
    # A common positive scale for the complete window avoids changing slow-time gain ratios.
    scale=abs(x).max((0,1),keepdims=True)
    scaled=np.divide(x,scale,out=np.zeros_like(x),where=scale>0)
    common=scaled.mean(1)
    a=common[1:];b=common[:-1];power=abs(a)**2+abs(b)**2
    q=np.divide(2*a*b.conj(),power,out=np.zeros_like(a),where=power>0)
    out[:,0,1:]=np.where(power>0,q.real-1,0).T;out[:,1,1:]=q.imag.T
    scale=abs(x).max(1,keepdims=True)
    u=np.divide(x,scale,out=np.zeros_like(x),where=scale>0)
    for k in range(1,8):
        a=u[:,k:];b=u[:,:-k]
        numerator=np.sum(a*b.conj(),axis=1)
        den=np.sqrt(np.sum(abs(a)**2,axis=1)*np.sum(abs(b)**2,axis=1))
        q=np.divide(numerator,den,out=np.zeros_like(numerator),where=den>0)
        out[:,2*k,:]=np.where(den>0,q.real-1,0).T;out[:,2*k+1,:]=q.imag.T
    return out


class BoundedChirpCorrelationReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([nn.Conv1d(16 if i==0 else 8,8,5) for i in range(5)])
        self.cell=nn.Linear(16,16)
        self.head=nn.Sequential(nn.Linear(32,16),nn.ELU(),nn.Linear(16,1))

    def forward(self,wave):
        n,c,ch,t=wave.shape;assert ch==16
        z=wave.reshape(n*c,16,t)
        for i,layer in enumerate(self.layers):
            z=torch.nn.functional.elu(circular_shift_convolution(z,layer.weight,layer.bias,2**i))
        mean=z.mean(-1);variance=(z-mean[:,:,None]).square().mean(-1)
        z=torch.nn.functional.elu(self.cell(torch.cat((mean,variance),1))).reshape(n,c,16)
        mean=z.mean(1);variance=(z-mean[:,None]).square().mean(1)
        return self.head(torch.cat((mean,variance),1))[:,0]


def numpy_forward(wave,state):
    st={k:v.detach().cpu().numpy().astype(float) for k,v in state.items()}
    n,c,ch,t=wave.shape;z=np.asarray(wave,float).reshape(n*c,ch,t)
    elu=lambda x:np.maximum(x,0)+np.expm1(np.minimum(x,0))
    for i in range(5):
        w=st[f'layers.{i}.weight'];bias=st[f'layers.{i}.bias'];d=2**i
        padded=np.pad(z,((0,0),(0,0),(2*d,2*d)),mode='wrap')
        result=np.broadcast_to(bias[None,:,None],(len(z),8,t)).copy()
        for k in range(5):result+=np.einsum('nit,oi->not',padded[:,:,k*d:k*d+t],w[:,:,k],optimize=True)
        z=elu(result)
    mean=np.sum(z,axis=-1)/t;variance=np.sum((z-mean[:,:,None])**2,axis=-1)/t
    z=elu(np.concatenate((mean,variance),axis=1)@st['cell.weight'].T+st['cell.bias']).reshape(n,c,16)
    mean=z.sum(1)/c;variance=np.sum((z-mean[:,None])**2,axis=1)/c
    z=elu(np.concatenate((mean,variance),axis=1)@st['head.0.weight'].T+st['head.0.bias'])
    return (z@st['head.2.weight'].T+st['head.2.bias'])[:,0]
