"""Zero prediction expert in train-standardized coordinates."""
from torch import nn


class ZeroField(nn.Module):
    def forward(self, x, m):
        return x * m
