"""Classical architectures + real PyPOTS implementation backbones, adapted to EEG reconstruction."""
import torch
from torch import nn
NAMES = ['Ridge', 'MLP_DAE', 'ConvDAE', 'UNet1D', 'BiLSTM_AE', 'TCN_AE', 'Transformer', 'GPVAE', 'SAITS', 'TimesNet', 'iTransformer', 'TimeMixer']
PYPOTS = ['Transformer', 'GPVAE', 'SAITS', 'TimesNet', 'iTransformer', 'TimeMixer']

def masked_mse(pred, target, observed):
    missing = 1 - observed
    return ((pred - target).square() * missing).sum() / missing.sum().clamp_min(1)

class MLPDAE(nn.Module):

    def __init__(self, c, t):
        super().__init__()
        self.c = c
        self.t = t
        self.net = nn.Sequential(nn.Linear(2 * c * t, 256), nn.GELU(), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 256), nn.GELU(), nn.Linear(256, c * t))

    def forward(self, x, m):
        return self.net(torch.cat([x, m], 1).flatten(1)).reshape(-1, self.c, self.t)

class ConvDAE(nn.Module):

    def __init__(self, c, t):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(2 * c, 64, 7, 2, 3), nn.GELU(), nn.Conv1d(64, 128, 5, 2, 2), nn.GELU(), nn.Conv1d(128, 128, 5, 2, 2), nn.GELU(), nn.ConvTranspose1d(128, 128, 4, 2, 1), nn.GELU(), nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.GELU(), nn.ConvTranspose1d(64, c, 4, 2, 1))

    def forward(self, x, m):
        return self.net(torch.cat([x, m], 1))

def convblock(a, b):
    return nn.Sequential(nn.Conv1d(a, b, 5, padding=2), nn.GroupNorm(8, b), nn.GELU(), nn.Conv1d(b, b, 3, padding=1), nn.GELU())

class UNet1D(nn.Module):

    def __init__(self, c, t):
        super().__init__()
        self.e1 = convblock(2 * c, 32)
        self.e2 = convblock(32, 64)
        self.mid = convblock(64, 128)
        self.d2 = convblock(128 + 64, 64)
        self.d1 = convblock(64 + 32, 32)
        self.out = nn.Conv1d(32, c, 1)

    def forward(self, x, m):
        a = self.e1(torch.cat([x, m], 1))
        b = self.e2(nn.functional.avg_pool1d(a, 2))
        h = self.mid(nn.functional.avg_pool1d(b, 2))
        h = self.d2(torch.cat([nn.functional.interpolate(h, size=b.shape[-1], mode='linear', align_corners=False), b], 1))
        h = self.d1(torch.cat([nn.functional.interpolate(h, size=a.shape[-1], mode='linear', align_corners=False), a], 1))
        return self.out(h)

class BiLSTMAE(nn.Module):

    def __init__(self, c, t):
        super().__init__()
        self.lstm = nn.LSTM(2 * c, 64, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(128, c)

    def forward(self, x, m):
        return self.out(self.lstm(torch.cat([x, m], 1).transpose(1, 2))[0]).transpose(1, 2)

class TCNBlock(nn.Module):

    def __init__(self, width, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.conv1 = nn.Conv1d(width, width, 3, dilation=dilation, padding=self.pad)
        self.conv2 = nn.Conv1d(width, width, 3, dilation=dilation, padding=self.pad)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        y = self.drop(nn.functional.gelu(self.conv1(x)[..., :-self.pad]))
        y = self.drop(nn.functional.gelu(self.conv2(y)[..., :-self.pad]))
        return x + y

class TCNAE(nn.Module):

    def __init__(self, c, t):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(2 * c, 64, 1), *[TCNBlock(64, 2 ** i) for i in range(6)], nn.Conv1d(64, c, 1))

    def forward(self, x, m):
        return self.net(torch.cat([x, m], 1))

class PyPOTSBackbone(nn.Module):

    def __init__(self, name, c, t):
        super().__init__()
        self.name = name
        self.t = t
        import pypots.imputation as imputation
        common = {'n_steps': t, 'n_features': c, 'device': 'cuda:0', 'verbose': False, 'saving_path': None}
        attention = {'n_layers': 2, 'd_model': 64, 'n_heads': 4, 'd_k': 16, 'd_v': 16, 'd_ffn': 128, 'dropout': 0.1}
        if name in ['Transformer', 'SAITS', 'iTransformer']:
            specific = attention
        elif name == 'GPVAE':
            specific = {'latent_size': 16, 'encoder_sizes': (64, 64), 'decoder_sizes': (64, 64), 'beta': 0.001, 'window_size': 3}
        elif name == 'TimesNet':
            specific = {'n_layers': 2, 'top_k': 3, 'd_model': 64, 'd_ffn': 128, 'n_kernels': 4, 'dropout': 0.1, 'apply_nonstationary_norm': False}
        elif name == 'TimeMixer':
            specific = {'n_layers': 2, 'd_model': 64, 'd_ffn': 128, 'top_k': 3, 'downsampling_layers': 2, 'downsampling_window': 2, 'channel_independence': False, 'apply_nonstationary_norm': False, 'dropout': 0.1}
        else:
            raise ValueError(name)
        obj = getattr(imputation, name)(**common, **specific)
        self.core = obj.model
        if name == 'GPVAE':
            from gpvae_numerics import StableGpvaeEncoder
            encoder = StableGpvaeEncoder(c, 16, (64, 64), 3).cuda()
            encoder.load_state_dict(self.core.backbone.encoder.state_dict())
            self.core.backbone.encoder = encoder
        self.configuration = {'library': 'PyPOTS 1.5', 'class': name, 'constructor': {**common, **specific}, 'adaptation': 'Supervised masked-channel reconstruction objective; original backbone, not claimed as exact original-paper training recipe.'}

    def inputs(self, x, m, target=None):
        d = {'X': x.transpose(1, 2), 'missing_mask': m.transpose(1, 2)}
        if target is not None:
            d.update({'X_ori': target.transpose(1, 2), 'indicating_mask': (1 - m).transpose(1, 2)})
        return d

    def forward(self, x, m, samples=1):
        if self.name == 'GPVAE':
            q = self.core.backbone.encode(x.transpose(1, 2))
            z = q.rsample() if self.training else q.mean
            out = self.core.backbone.decode(z).mean
        else:
            out = self.core(self.inputs(x, m))['reconstruction']
        return out.transpose(1, 2)

    def loss(self, x, m, target):
        if self.name == 'GPVAE':
            backbone = self.core.backbone
            q = backbone.encode(x.transpose(1, 2))
            pred = backbone.decode(q.rsample()).mean.transpose(1, 2)
            loss = masked_mse(pred, target, m)
            if backbone.prior is None:
                backbone.prior = backbone._init_prior(device=x.device)
            kl = torch.distributions.kl_divergence(q, backbone.prior).mean() / self.t
            return loss + 0.001 * kl
        return masked_mse(self(x, m), target, m)

def make_model(name, c, t):
    if name in PYPOTS:
        return PyPOTSBackbone(name, c, t)
    cls = {'MLP_DAE': MLPDAE, 'ConvDAE': ConvDAE, 'UNet1D': UNet1D, 'BiLSTM_AE': BiLSTMAE, 'TCN_AE': TCNAE}.get(name)
    if cls is None:
        raise ValueError(name)
    return cls(c, t)
