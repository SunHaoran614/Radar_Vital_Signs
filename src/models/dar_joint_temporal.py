"""Learn temporal features of cross-cell moments before time pooling."""
import numpy as np
import torch
from torch import nn
from src.models.dar_circular_temporal import CircularTemporalReader


class JointTemporalReader(CircularTemporalReader):
    def __init__(self):
        super().__init__('circular')
        self.joint=nn.Conv1d(16,8,5,padding=2,padding_mode='circular')

    def forward(self,x):
        n,c,t=x.shape;z=x.reshape(n*c,1,t)
        for layer in self.layers:z=torch.nn.functional.elu(layer(z))
        z=z.reshape(n,c,8,t)
        mean=z.mean(1);variance=(z-mean[:,None]).square().mean(1)
        z=torch.nn.functional.elu(self.joint(torch.cat((mean,variance),1)))
        return self.head(z.mean(-1))[:,0]


def numpy_forward(x,state):
    st={k:v.detach().numpy().astype(float) for k,v in state.items()}
    n,c,t=x.shape;z=x.astype(float).reshape(n*c,1,t)
    def elu(a):return np.where(a>0,a,np.expm1(np.minimum(a,0)))
    def conv(a,w,bias,dilation=1):
        pad=dilation*(w.shape[-1]//2)
        padded=np.pad(a,((0,0),(0,0),(pad,pad)),mode='wrap')
        output=np.broadcast_to(bias[None,:,None],(len(a),len(bias),a.shape[-1])).copy()
        for k in range(w.shape[-1]):
            output+=np.einsum('nit,oi->not',padded[:,:,k*dilation:k*dilation+a.shape[-1]],w[:,:,k],optimize=True)
        return output
    for i in range(4):z=elu(conv(z,st[f'layers.{i}.weight'],st[f'layers.{i}.bias'],2**i))
    z=z.reshape(n,c,8,t);mean=z.mean(1)
    # Direct second central moment rather than subtracting two close powers.
    variance=np.sum((z-mean[:,None])**2,axis=1)/c
    z=elu(conv(np.concatenate((mean,variance),axis=1),st['joint.weight'],st['joint.bias'])).mean(-1)
    z=elu(z@st['head.0.weight'].T+st['head.0.bias'])
    return (z@st['head.2.weight'].T+st['head.2.bias'])[:,0]
