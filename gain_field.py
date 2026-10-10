"""Global gain transport and normalized spectral field."""
import torch
from torch import nn
from field import BANDS

@torch.no_grad()
def fit_gain_fields(z, train):
    c = z.shape[1]
    cov = torch.zeros(5, c, c, device=z.device, dtype=torch.float64)
    logs = [[] for _ in BANDS]
    freq = torch.fft.rfftfreq(z.shape[-1], 1 / 128).to(z.device)
    for s in range(0, len(train), 128):
        ft = torch.fft.rfft(z[torch.as_tensor(train[s:s + 128], device=z.device)]).to(torch.complex128)
        for j, (lo, hi) in enumerate(BANDS):
            f = ft[:, :, (freq >= lo) & (freq < hi)]
            g = f.abs().square().mean(-1).sqrt().clamp_min(1e-06)
            unit = f / g[:, :, None]
            cov[j] += torch.einsum('ncf,ndf->cd', unit, unit.conj()).real / f.shape[-1]
            logs[j].append(g.log())
    cov /= len(train)
    means = []
    amplitudes = []
    for values in logs:
        a = torch.cat(values)
        mu = a.mean(0)
        delta = a - mu
        k = delta.T @ delta / len(a)
        k /= k.diagonal().mean().clamp_min(1e-08)
        means.append(mu)
        amplitudes.append(k)
    return {'correlation': cov.float(), 'gain_mean': torch.stack(means).float(), 'gain_covariance': torch.stack(amplitudes).float()}

class GainTransport(nn.Module):

    def __init__(self, priors):
        super().__init__()
        for name, value in priors.items():
            self.register_buffer(name, value)
        self.lambdas = (0.01, 0.1, 1.0, 10.0)
        c = self.correlation.shape[-1]
        self.channel_embedding = nn.Parameter(torch.randn(c, 8) * 0.02)
        self.calibrators = nn.ModuleList([nn.Sequential(nn.Linear(15, 32), nn.GELU(), nn.Linear(32, 6)) for _ in BANDS])
        for net in self.calibrators:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            net[-1].bias.data[1] = 2
            net[-1].bias.data[-1] = 0.693147

    def forward(self, x, m):
        x = x * m
        b, c, t = x.shape
        v = m[:, :, 0]
        ft = torch.fft.rfft(x)
        freq = torch.fft.rfftfreq(t, 1 / 128).to(x.device)
        result = torch.zeros_like(ft)
        for j, (lo, hi) in enumerate(BANDS):
            use = (freq >= lo) & (freq < hi)
            f = ft[:, :, use]
            real = torch.cat([f.real, f.imag], -1)
            with torch.no_grad():
                g = f.abs().square().mean(-1).sqrt().clamp_min(1e-06)
                logg = g.log()
                center = (logg - self.gain_mean[j]) * v
                n = v.sum(1, keepdim=True)
                shift = center.sum(1, keepdim=True) / n
                logpred = self.gain_mean[j] + shift
                logloo = self.gain_mean[j] + (center.sum(1, keepdim=True) - center) / (n - 1).clamp_min(1)
                gain = logpred.clamp(-12, 12).exp()
                loo_ratio = (logloo - logg).clamp(-12, 12).exp()
                unit = real / g[:, :, None]
                k = self.correlation[j]
                experts = []
                scores = []
                for lam in self.lambdas:
                    a = k[None] * v[:, :, None] * v[:, None, :] + torch.diag_embed(1 - v + v * lam)
                    p = torch.linalg.inv(a)
                    beta = p @ unit
                    pred = k[None] @ (beta * v[:, :, None])
                    experts.append(pred * gain[:, :, None])
                    loo = unit - beta / p.diagonal(dim1=-2, dim2=-1)[:, :, None].clamp_min(1e-08)
                    error = (loo * loo_ratio[:, :, None] - unit).square().sum(-1) / unit.square().sum(-1).clamp_min(1e-08)
                    scores.append((error * v).sum(1) / n.squeeze(1))
                scores = torch.stack(scores, 1).clamp(1e-06, 1000000.0).log()
                power = (real.square().sum((1, 2)) / n.squeeze(1) / real.shape[-1]).clamp_min(1e-08).log()[:, None]
                fraction = v.mean(1)[:, None]
            features = torch.cat([scores, power, fraction], 1)[:, None].expand(-1, c, -1)
            features = torch.cat([features, self.channel_embedding[None].expand(b, -1, -1), torch.ones(b, c, 1, device=x.device)], -1)
            controls = self.calibrators[j](features)
            weights = torch.softmax(controls[:, :, :5], -1)
            scale = 1.5 * torch.sigmoid(controls[:, :, -1:])
            experts.append(torch.zeros_like(experts[0]))
            out = (torch.stack(experts, 2) * weights[:, :, :, None]).sum(2) * scale
            n_freq = f.shape[-1]
            result[:, :, use] = torch.complex(out[:, :, :n_freq], out[:, :, n_freq:])
        return x + torch.fft.irfft(result, n=t) * (1 - m)

def exact_relative_loss(pred, target, mask):
    energy = target.double().square().sum(-1).clamp_min(1e-12)
    error = (pred.double() - target.double()).square().sum(-1)
    missing = 1 - mask[:, :, 0]
    return (error / energy * missing).sum() / missing.sum().clamp_min(1)
