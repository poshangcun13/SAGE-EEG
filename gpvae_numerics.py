"""Equivalent banded precision construction with exact triangular zeros.

Based on PyPOTS 1.5 GpvaeEncoder (BSD-3-Clause; Jun Wang and Wenjie Du).
CUDA triangular solves can leave subnormal entries above the theoretical
lower-triangular covariance factor. Explicit tril restores that invariant.
"""
import torch
from pypots.nn.modules.gpvae.layers import GpvaeEncoder

class StableGpvaeEncoder(GpvaeEncoder):

    def forward(self, x):
        mapped = self.net(x)
        batch, t = mapped.shape[:2]
        mean = self.mu_layer(mapped).transpose(-1, -2)
        covariance = torch.sigmoid(self.logvar_layer(mapped).transpose(-1, -2)).reshape(batch, self.z_size, 2 * t)
        precision = torch.diag_embed(covariance[..., :t] + 1) + torch.diag_embed(covariance[..., t:-1], offset=1)
        identity = torch.eye(t, device=mapped.device, dtype=mapped.dtype).expand(batch, self.z_size, t, t)
        factor = torch.linalg.solve_triangular(precision, identity, upper=True)
        factor = torch.where(torch.isfinite(factor), factor, torch.zeros_like(factor))
        return torch.distributions.MultivariateNormal(mean, scale_tril=factor.transpose(-1, -2).tril())
