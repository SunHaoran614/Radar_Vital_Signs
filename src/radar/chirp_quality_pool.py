"""Shared quality weights for coherent chirp-group ROI aggregation.

This component accepts [batch, cell, group, time] complex observations. It
does not identify heartbeat, calibrate inter-group phase, or consume labels.
"""
import torch
from torch import nn


class ChirpQualityPool(nn.Module):
    """Convex complex pooling with an identity contract for a single group."""

    def __init__(self):
        super().__init__()
        self.features = nn.Conv1d(3, 8, 5, padding=2)
        self.score = nn.Conv1d(8, 1, 1, bias=False)
        nn.init.zeros_(self.score.weight)

    def forward(self, x, priors=None, learned=True):
        if x.ndim != 4 or not x.is_complex() or min(x.shape) < 1:
            raise ValueError('Expected nonempty complex [batch,cell,group,time]')
        n, c, g, t = x.shape
        if priors is None:
            prior = torch.ones(g, dtype=x.real.dtype, device=x.device) / g
        else:
            prior = torch.as_tensor(priors, dtype=x.real.dtype, device=x.device)
            if prior.shape != (g,) or not torch.isfinite(prior).all() or not (prior > 0).all():
                raise ValueError('One finite positive prior per group required')
            prior = prior / prior.sum()
        prior = prior.reshape(1, 1, g, 1)
        reference = (x * prior).sum(2, keepdim=True)
        energy = (x.abs().square() * prior).sum(2, keepdim=True).mean(-1, keepdim=True)
        norm = energy.clamp_min(1e-12).sqrt()
        log_amplitude = (x.abs() / norm).clamp_min(1e-6).log().clamp(-10, 10)
        denominator = (reference.abs().square() + 1e-4 * energy).clamp_min(1e-12)
        deviation = (x - reference) * reference.conj() / denominator
        z = torch.stack((log_amplitude, deviation.real.clamp(-10, 10),
                         deviation.imag.clamp(-10, 10)), dim=3)
        z = z.reshape(n*c*g, 3, t)
        logits = self.score(torch.nn.functional.elu(self.features(z))).reshape(n, c, g, t)
        if not learned:
            logits = logits * 0
        unnormalized = prior * (logits - logits.amax(2, keepdim=True)).exp()
        weights = unnormalized / unnormalized.sum(2, keepdim=True)
        return (x * weights).sum(2), weights
