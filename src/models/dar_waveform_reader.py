"""Three-view waveform reader used by the fixed DAR inference pipeline.

The checkpoint parameter names and numerical operations match the original
research implementation. No experiment runners or reference labels are loaded.
"""
import torch
from torch import nn

GRID = torch.arange(10, 45, dtype=torch.float32) * 4.6875

def distribution(wave):
    z=wave-wave.mean(-1,keepdim=True);z=z/z.square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()
    return (torch.fft.rfft(z,norm='forward').abs().square()[:,10:45]*128).softmax(-1)


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders=nn.ModuleList([nn.Sequential(nn.Conv1d(1,4,5,padding=2),nn.ELU(),nn.Conv1d(4,8,5,padding=2),nn.ELU()) for _ in range(3)])
        self.decoder=nn.Sequential(nn.Conv1d(8,4,3,padding=1),nn.ELU(),nn.Conv1d(4,1,3,padding=1));self.register_buffer('primary',torch.tensor(0,dtype=torch.int64))
    def wave(self,x,index):
        n,c,t=x.shape;z=self.encoders[index](x.reshape(n*c,1,t)).reshape(n,c,8,t).mean(1)
        return self.decoder(z)[:,0]
    def probabilities(self,x):return distribution(self.wave(x,int(self.primary)))
    def forward(self,x):return self.probabilities(x)@GRID
    def clone_source(self):
        for i in (1,2):self.encoders[i].load_state_dict(self.encoders[0].state_dict())
