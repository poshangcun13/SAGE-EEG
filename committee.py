"""SAGE-EEG: heterogeneous fields with equal-subject convex risk calibration."""
import torch
from torch import nn
from field import AbsoluteField
from gain_field import GainTransport
from library import WitnessLibrary
from zero_expert import ZeroField

def build_experts(absolute, gain, library):
    a = AbsoluteField(absolute['covariance'], absolute['bands'])
    a.load_state_dict(absolute['state_dict'])
    g = GainTransport(gain['priors'])
    g.load_state_dict(gain['state_dict'])
    return [a, g, WitnessLibrary(library['covariance'], library['bands'], lam=0.1, temperature=1.0, shrink=True), ZeroField()]

class SAGEEEG(nn.Module):

    def __init__(self, experts, weights, channels):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.register_buffer('weights', weights.float())
        self.register_buffer('ratios', torch.tensor([int(channels * r) / channels for r in [0.25, 0.5, 0.75]], device=weights.device))

    def forward(self, x, m):
        x = x * m
        ratio = 1 - m[:, :, 0].mean(1)
        which = (ratio[:, None] - self.ratios[None]).abs().argmin(1)
        w = self.weights[which]
        p = torch.zeros_like(x)
        for j, expert in enumerate(self.experts):
            if bool(torch.any(w[:, j] > 1e-07)):
                p = p + expert(x, m) * w[:, j, None, None]
        return x + p * (1 - m)
