"""Equivalent circular dilated convolution using explicit shifted matrix products.

Separate implementation: existing frozen training/model code remains untouched.
"""
import torch
from src.models.dar_joint_temporal import JointTemporalReader


def circular_shift_convolution(x, weight, bias, dilation):
    radius = weight.shape[-1]//2
    result = None
    for k in range(weight.shape[-1]):
        shifted = torch.roll(x, shifts=dilation*(radius-k), dims=-1)
        value = torch.einsum('nit,oi->not', shifted, weight[:, :, k])
        result = value if result is None else result+value
    return result+bias[None, :, None]


class MatmulJointTemporalReader(JointTemporalReader):
    def forward(self, x):
        n,c,t = x.shape
        z = x.reshape(n*c,1,t)
        for i,layer in enumerate(self.layers):
            z = torch.nn.functional.elu(circular_shift_convolution(z,layer.weight,layer.bias,2**i))
        z = z.reshape(n,c,8,t)
        mean = z.mean(1)
        variance = (z-mean[:,None]).square().mean(1)
        z = torch.nn.functional.elu(self.joint(torch.cat((mean,variance),1)))
        return self.head(z.mean(-1))[:,0]
