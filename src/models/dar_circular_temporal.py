"""Shared dilated temporal reader with explicit finite-window shift invariance."""
import numpy as np
import torch
from torch import nn


class CircularTemporalReader(nn.Module):
    def __init__(self, padding='circular'):
        super().__init__()
        if padding not in ('circular','zeros'):raise ValueError(padding)
        self.padding=padding
        channels=(1,4,8,8,8)
        self.layers=nn.ModuleList([nn.Conv1d(channels[i],channels[i+1],5,
            dilation=2**i,padding=2*2**i,padding_mode=padding) for i in range(4)])
        self.head=nn.Sequential(nn.Linear(8,16),nn.ELU(),nn.Linear(16,1))

    def forward(self,x):
        n,c,t=x.shape
        z=x.reshape(n*c,1,t)
        for layer in self.layers:z=torch.nn.functional.elu(layer(z))
        # Average time separately for each cell; never mix unaligned waveforms.
        return self.head(z.mean(-1).reshape(n,c,8).mean(1))[:,0]


def numpy_forward(x,state,padding):
    st={k:v.detach().numpy().astype(float) for k,v in state.items()}
    n,c,t=x.shape;z=x.astype(float).reshape(n*c,1,t)
    def elu(a):return np.where(a>0,a,np.expm1(np.minimum(a,0)))
    for i in range(4):
        w=st[f'layers.{i}.weight'];bias=st[f'layers.{i}.bias'];d=2**i
        padded=np.pad(z,((0,0),(0,0),(2*d,2*d)),mode='wrap' if padding=='circular' else 'constant')
        out=np.broadcast_to(bias[None,:,None],(len(z),len(bias),t)).copy()
        for k in range(5):out+=np.einsum('nit,oi->not',padded[:,:,k*d:k*d+t],w[:,:,k],optimize=True)
        z=elu(out)
    z=z.mean(-1).reshape(n,c,8).mean(1)
    z=elu(z@st['head.0.weight'].T+st['head.0.bias'])
    return (z@st['head.2.weight'].T+st['head.2.bias'])[:,0]
