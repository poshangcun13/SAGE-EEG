"""Visible-witness retrieval over training-subject field archetypes."""
import torch
from torch import nn
from field import estimate_fields

@torch.no_grad()
def fit_library(z, train, subjects):
    fields = []
    for subject in sorted(set(subjects[train])):
        ids = train[subjects[train] == subject]
        cov, bands = estimate_fields(z, ids, 'robust', True)
        fields.append(cov)
    return (torch.stack(fields, 1), bands)

class WitnessLibrary(nn.Module):

    def __init__(self, cov, bands, lam=0.1, temperature=0.2, shrink=True):
        super().__init__()
        self.register_buffer('covariance', cov)
        self.bands = bands
        self.lam = lam
        self.temperature = temperature
        self.shrink = shrink

    def forward(self, x, m):
        x = x * m
        b, c, t = x.shape
        v = m[:, :, 0]
        ft = torch.fft.rfft(x)
        freq = torch.fft.rfftfreq(t, 1 / 128).to(x.device)
        out = torch.zeros_like(ft)
        for j, (lo, hi) in enumerate(self.bands):
            use = (freq >= lo) & (freq < hi)
            f = ft[:, :, use]
            real = torch.cat([f.real, f.imag], -1)
            k = self.covariance[j]
            a = k[None] * v[:, None, :, None] * v[:, None, None, :] + torch.diag_embed(1 - v + v * self.lam)[:, None]
            p = torch.linalg.inv(a)
            coef = p @ real[:, None]
            pred = k[None] @ (coef * v[:, None, :, None])
            loo = real[:, None] - coef / p.diagonal(dim1=-2, dim2=-1)[:, :, :, None].clamp_min(1e-08)
            energy = real.square().sum(-1).clamp_min(1e-08)
            if self.shrink:
                cross = ((loo * real[:, None]).sum(-1) / energy[:, None] * v[:, None]).sum(-1)
                power = (loo.square().sum(-1) / energy[:, None] * v[:, None]).sum(-1)
                gain = (cross / power.clamp_min(1e-08)).clamp(0, 1.5)
            else:
                gain = torch.ones(b, k.shape[0], device=x.device)
            error = ((loo * gain[:, :, None, None] - real[:, None]).square().sum(-1) / energy[:, None] * v[:, None]).sum(-1) / v.sum(1)[:, None]
            score = torch.cat([error, torch.ones(b, 1, device=x.device)], 1)
            weights = torch.softmax(-score / self.temperature, 1)
            pred = (pred * gain[:, :, None, None] * weights[:, :-1, None, None]).sum(1)
            n = f.shape[-1]
            out[:, :, use] = torch.complex(pred[:, :, :n], pred[:, :, n:])
        return x + torch.fft.irfft(out, n=t) * (1 - m)
