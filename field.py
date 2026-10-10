"""Absolute-amplitude field and visible-channel risk calibration."""
import torch
from torch import nn
BANDS = [(0, 4), (4, 8), (8, 13), (13, 30), (30, 65)]

@torch.no_grad()
def estimate_fields(z, train, mode='robust', spectral=True):
    c = z.shape[1]
    bands = BANDS if spectral else [(0, 65)]
    cov = torch.zeros(len(bands), c, c, device=z.device, dtype=torch.float64)
    weight = 0
    frequencies = torch.fft.rfftfreq(z.shape[-1], 1 / 128).to(z.device)
    for start in range(0, len(train), 128):
        y = z[torch.as_tensor(train[start:start + 128], device=z.device)]
        w = 1 / y.square().mean((1, 2)).clamp_min(1e-05) if mode == 'robust' else torch.ones(len(y), device=z.device)
        ft = torch.fft.rfft(y).to(torch.complex128)
        for j, (lo, hi) in enumerate(bands):
            f = ft[:, :, (frequencies >= lo) & (frequencies < hi)]
            cov[j] += torch.einsum('ncf,ndf,n->cd', f, f.conj(), w.to(torch.complex128)).real / max(1, f.shape[-1])
        weight += float(w.sum())
    cov /= weight
    cov = cov / cov.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None].clamp_min(1e-12)
    return (cov.float(), bands)

class AbsoluteField(nn.Module):
    """Absolute-amplitude field and visible-channel risk calibration."""

    def __init__(self, cov, bands):
        super().__init__()
        self.register_buffer('covariance', cov)
        self.bands = bands
        self.lambdas = (0.01, 0.1, 1.0, 10.0)
        c = cov.shape[-1]
        self.channel_embedding = nn.Parameter(torch.randn(c, 8) * 0.02)
        self.calibrators = nn.ModuleList([nn.Sequential(nn.Linear(15, 32), nn.GELU(), nn.Linear(32, 5)) for _ in bands])
        for net in self.calibrators:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            net[-1].bias.data[1] = 2
            net[-1].bias.data[-1] = 0.693147

    def forward(self, x, m):
        x = x * m
        b, c, t = x.shape
        visible = m[:, :, 0]
        ft = torch.fft.rfft(x)
        freq = torch.fft.rfftfreq(t, 1 / 128).to(x.device)
        reconstructed = torch.zeros_like(ft)
        for j, (lo, hi) in enumerate(self.bands):
            use = (freq >= lo) & (freq < hi)
            f = ft[:, :, use]
            real = torch.cat([f.real, f.imag], -1)
            k = self.covariance[j]
            experts = []
            scores = []
            with torch.no_grad():
                for lam in self.lambdas:
                    a = k[None] * visible[:, :, None] * visible[:, None, :] + torch.diag_embed(1 - visible + visible * lam)
                    precision = torch.linalg.inv(a)
                    coefficients = precision @ real
                    experts.append(k[None] @ (coefficients * visible[:, :, None]))
                    residual = coefficients / precision.diagonal(dim1=-2, dim2=-1)[:, :, None].clamp_min(1e-08)
                    error = residual.square().sum(-1) / real.square().sum(-1).clamp_min(1e-08)
                    scores.append((error * visible).sum(1) / visible.sum(1))
                score = torch.stack(scores, 1).clamp(1e-06, 1000000.0).log()
                power = (real.square().sum((1, 2)) / visible.sum(1) / real.shape[-1]).clamp_min(1e-08).log()[:, None]
                fraction = visible.mean(1)[:, None]
            features = torch.cat([score, power, fraction], 1)[:, None].expand(-1, c, -1)
            features = torch.cat([features, self.channel_embedding[None].expand(b, -1, -1), torch.ones(b, c, 1, device=x.device)], -1)
            controls = self.calibrators[j](features)
            weights = torch.softmax(controls[:, :, :4], -1)
            gain = 1.5 * torch.sigmoid(controls[:, :, -1:])
            out = (torch.stack(experts, 2) * weights[:, :, :, None]).sum(2) * gain
            n = f.shape[-1]
            reconstructed[:, :, use] = torch.complex(out[:, :, :n], out[:, :, n:])
        pred = torch.fft.irfft(reconstructed, n=t)
        return x * m + pred * (1 - m)

def relative_loss(pred, target, mask):
    error = (pred - target).square().sum(-1)
    energy = target.square().sum(-1).clamp_min(1e-08)
    missing = 1 - mask[:, :, 0]
    return (error / energy * missing).sum() / missing.sum().clamp_min(1)
